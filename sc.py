import asyncio
import aiohttp
import json
import time
import math
import pandas as pd
import numpy as np
from datetime import datetime
from rich.live import Live
from rich.layout import Layout
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# ==========================================
# CONFIGURACIÓN GLOBAL Y CONSTANTES
# ==========================================
BASE_REST_URL = "https://api.binance.com"
BASE_WS_URL = "wss://stream.binance.com:9443/stream?streams="
TOP_N_PAIRS = 200
RSI_PERIOD = 14
BB_PERIOD = 20
BB_STD = 2

# ==========================================
# ESTADO DEL SISTEMA (In-Memory Store)
# ==========================================
class ScannerState:
    def __init__(self):
        self.pairs = []
        self.tick_sizes = {}
        self.klines_1h = {}   
        self.klines_15m = {}  
        self.klines_5m = {}   
        
        self.radar = {} 
        self.active_signals = {}
        
        self.ws_connection = None 
        self.activity_msg = "Iniciando sistema..."
        self.ping = 0
        self.start_time = time.time()

state = ScannerState()

# ==========================================
# MOTOR MATEMÁTICO (TradingView Exact Match)
# ==========================================
def rma_tradingview(series: pd.Series, period: int) -> pd.Series:
    alpha = 1 / period
    rma = series.copy()
    rma.iloc[period] = series.iloc[1:period+1].mean()
    for i in range(period+1, len(series)):
        rma.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * rma.iloc[i-1]
    return rma

def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)
    avg_gain = rma_tradingview(gain, period)
    avg_loss = rma_tradingview(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(series: pd.Series, period: int = BB_PERIOD, std_dev: int = BB_STD):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std(ddof=0)
    upper_bb = sma + (std_dev * std)
    lower_bb = sma - (std_dev * std)
    return sma, upper_bb, lower_bb

def get_indicators(df: pd.DataFrame):
    """Retorna las SERIES COMPLETAS para evaluación de la vela actual y anteriores."""
    if len(df) < max(BB_PERIOD, 50): 
        return None, None, None
    rsi = calculate_rsi(df['close'])
    _, upper_bb, lower_bb = calculate_bollinger_bands(df['close'])
    return rsi, upper_bb, lower_bb

# ==========================================
# CONEXIÓN A EXCHANGES Y DATOS REST
# ==========================================
async def fetch_json(session: aiohttp.ClientSession, url: str, params: dict = None):
    start = time.time()
    async with session.get(url, params=params) as response:
        state.ping = int((time.time() - start) * 1000)
        return await response.json()

async def initialize_pairs(session: aiohttp.ClientSession):
    state.activity_msg = "[Recopilando Exchange Info y Top Pares...]"
    
    # 1. Obtener Exchange Info para tick sizes (precisión) validando solo pares USDT
    exch_info = await fetch_json(session, f"{BASE_REST_URL}/api/v3/exchangeInfo")
    valid_binance_pairs = {}
    for symbol_data in exch_info['symbols']:
        if symbol_data['status'] == 'TRADING' and symbol_data['symbol'].endswith("USDT"):
            sym = symbol_data['symbol']
            for f in symbol_data['filters']:
                if f['filterType'] == 'PRICE_FILTER':
                    tick_size = float(f['tickSize'])
                    decimals = max(0, int(round(-math.log10(tick_size))))
                    valid_binance_pairs[sym] = {'tick': tick_size, 'decimals': decimals}

    selected_pairs = []
    
    # 2. Prioridad 1: Top Global por Capitalización de Mercado (API CoinGecko Pública)
    # Esto asegura que los pares sean fuertes y existan en Binance, OKX, Bybit, etc.
    state.activity_msg = "[Consultando Market Cap Global (CoinGecko)...]"
    try:
        cg_url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=1"
        async with session.get(cg_url) as resp:
            if resp.status == 200:
                cg_data = await resp.json()
                for coin in cg_data:
                    sym = f"{coin['symbol'].upper()}USDT"
                    if sym in valid_binance_pairs and sym not in selected_pairs:
                        selected_pairs.append(sym)
    except Exception:
        pass # Fallback silencioso si el servidor rechaza la conexión
        
    # 3. Prioridad 2 / Fallback: Completar con Top por Volumen 24h
    state.activity_msg = "[Completando Top con Volumen 24h...]"
    if len(selected_pairs) < TOP_N_PAIRS:
        tickers = await fetch_json(session, f"{BASE_REST_URL}/api/v3/ticker/24hr")
        vol_pairs = [{'symbol': t['symbol'], 'vol': float(t['quoteVolume'])} for t in tickers if t['symbol'] in valid_binance_pairs]
        vol_pairs.sort(key=lambda x: x['vol'], reverse=True)
        
        for p in vol_pairs:
            if p['symbol'] not in selected_pairs:
                selected_pairs.append(p['symbol'])
            if len(selected_pairs) >= TOP_N_PAIRS:
                break

    # Asignar pares finales y limpiar estado de precisiones en memoria
    state.pairs = selected_pairs[:TOP_N_PAIRS]
    state.tick_sizes = {sym: valid_binance_pairs[sym] for sym in state.pairs}

async def fetch_historical_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 500):
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    data = await fetch_json(session, f"{BASE_REST_URL}/api/v3/klines", params)
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    df['close'] = df['close'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    return df

async def sync_klines(session: aiohttp.ClientSession, interval: str, target_dict: dict, pairs_list: list):
    state.activity_msg = f"[Sincronizando Velas Históricas {interval}...]"
    chunk_size = 20
    for i in range(0, len(pairs_list), chunk_size):
        chunk = pairs_list[i:i+chunk_size]
        tasks = [fetch_historical_klines(session, sym, interval) for sym in chunk]
        results = await asyncio.gather(*tasks)
        for sym, df in zip(chunk, results):
            target_dict[sym] = df
        await asyncio.sleep(0.5)

# ==========================================
# ANÁLISIS DE MICROESTRUCTURA (ORDER BOOK)
# ==========================================
async def analyze_microstructure(session: aiohttp.ClientSession, symbol: str, side: str, ref_price: float):
    params = {'symbol': symbol, 'limit': 100}
    depth = await fetch_json(session, f"{BASE_REST_URL}/api/v3/depth", params)
    
    tick_info = state.tick_sizes[symbol]
    tick_sz = tick_info['tick']
    decimals = tick_info['decimals']
    
    best_price = ref_price
    max_vol = 0.0

    if side == 'LONG':
        min_price = ref_price * 0.995
        for bid in depth['bids']:
            price = float(bid[0])
            qty = float(bid[1])
            vol = price * qty
            if min_price <= price <= ref_price * 1.005:
                if vol > max_vol:
                    max_vol = vol
                    best_price = price + tick_sz
                    
    elif side == 'SHORT':
        max_price = ref_price * 1.005
        for ask in depth['asks']:
            price = float(ask[0])
            qty = float(ask[1])
            vol = price * qty
            if ref_price * 0.995 <= price <= max_price:
                if vol > max_vol:
                    max_vol = vol
                    best_price = price - tick_sz

    formatted_price = f"{best_price:.{decimals}f}"
    return formatted_price, max_vol

# ==========================================
# LÓGICA CORE DIRECTA (DIRECTIVAS ESTRICTAS + MEMORIA VISUAL)
# ==========================================
async def evaluate_logic(session: aiohttp.ClientSession, symbol: str):
    df_1h = state.klines_1h.get(symbol)
    df_15m = state.klines_15m.get(symbol)
    df_5m = state.klines_5m.get(symbol)
    
    if df_1h is None or df_15m is None or df_5m is None:
        return

    ind_1h = get_indicators(df_1h)
    if ind_1h[0] is None: return
    rsi_1h_s, upper_1h_s, lower_1h_s = ind_1h

    ind_15m = get_indicators(df_15m)
    if ind_15m[0] is None: return
    rsi_15m_s, upper_15m_s, lower_15m_s = ind_15m

    ind_5m = get_indicators(df_5m)
    if ind_5m[0] is None: return
    rsi_5m_s, upper_5m_s, lower_5m_s = ind_5m
        
    # Extraemos valores actuales de la última vela (milisegundo actual) para la Interfaz
    rsi_1h_val = rsi_1h_s.iloc[-1]
    rsi_15m_val = rsi_15m_s.iloc[-1]
    rsi_5m_val = rsi_5m_s.iloc[-1]
    
    upper_5m_val = upper_5m_s.iloc[-1]
    lower_5m_val = lower_5m_s.iloc[-1]
    current_price = df_5m['close'].iloc[-1]

    # --- DIRECTIVAS MACRO (1H + 15m) CON MEMORIA DE ESTRUCTURA ---
    # El trader ve la "zona" de sobrecompra/sobreventa basada en el movimiento reciente.
    # Si exigimos que el milisegundo actual tenga RSI extremo, perdemos la entrada si el RSI ya empezó a rebotar.
    
    # 1H: Buscamos si en las últimas 3 velas (3 horas) cruzó la banda y tuvo RSI extremo
    long_1h = any((rsi_1h_s.iloc[-i] < 30 and df_1h['low'].iloc[-i] <= lower_1h_s.iloc[-i]) for i in range(1, 4) if len(df_1h) >= i)
    short_1h = any((rsi_1h_s.iloc[-i] > 70 and df_1h['high'].iloc[-i] >= upper_1h_s.iloc[-i]) for i in range(1, 4) if len(df_1h) >= i)
    
    # 15m: Buscamos si en los últimas 4 velas (1 hora) cruzó la banda y tuvo RSI extremo
    long_15m = any((rsi_15m_s.iloc[-i] < 30 and df_15m['low'].iloc[-i] <= lower_15m_s.iloc[-i]) for i in range(1, 5) if len(df_15m) >= i)
    short_15m = any((rsi_15m_s.iloc[-i] > 70 and df_15m['high'].iloc[-i] >= upper_15m_s.iloc[-i]) for i in range(1, 5) if len(df_15m) >= i)

    macro_side = None
    if long_1h and long_15m:
        macro_side = 'LONG'
    elif short_1h and short_15m:
        macro_side = 'SHORT'

    if macro_side:
        dist_pct = ((current_price - lower_5m_val) / lower_5m_val) * 100 if macro_side == 'LONG' else ((upper_5m_val - current_price) / upper_5m_val) * 100
        state.radar[symbol] = {
            'side': macro_side, 'rsi_1h': rsi_1h_val, 'rsi_15m': rsi_15m_val,
            'rsi_5m': rsi_5m_val, 'dist': dist_pct, 'price': current_price,
            'upper_5m': upper_5m_val, 'lower_5m': lower_5m_val
        }
    else:
        state.radar.pop(symbol, None)

    # --- DIRECTIVAS DE GATILLO (5m) ---
    if symbol in state.radar:
        radar_info = state.radar[symbol]
        trigger = False
        ref_price = current_price
        bb_value_raw = 0.0
        
        if radar_info['side'] == 'LONG':
            # Directiva 5m Long: RSI < 30 + Cruza Banda Inferior (Evaluamos vela actual y la anterior inmediata)
            if any((rsi_5m_s.iloc[-i] < 30 and df_5m['low'].iloc[-i] <= lower_5m_s.iloc[-i]) for i in range(1, 3) if len(df_5m) >= i):
                trigger = True
                ref_price = lower_5m_val
                bb_value_raw = lower_5m_val
                
        elif radar_info['side'] == 'SHORT':
            # Directiva 5m Short: RSI > 70 + Cruza Banda Superior (Evaluamos vela actual y la anterior inmediata)
            if any((rsi_5m_s.iloc[-i] > 70 and df_5m['high'].iloc[-i] >= upper_5m_s.iloc[-i]) for i in range(1, 3) if len(df_5m) >= i):
                trigger = True
                ref_price = upper_5m_val
                bb_value_raw = upper_5m_val

        # Solo si se cumplen las 3 directivas, consultamos el libro de órdenes
        if trigger:
            if symbol not in state.active_signals:
                entry_price, vol = await analyze_microstructure(session, symbol, radar_info['side'], ref_price)
                state.active_signals[symbol] = {
                    'side': radar_info['side'],
                    'bb_value': bb_value_raw, 
                    'entry': float(entry_price),
                    'wall_vol': vol,
                    'timestamp': time.time(),
                    'status': '¡GATILLO EN VIVO!',
                    'pnl': 0.0
                }
            else:
                # Actualiza el timestamp si la entrada sigue siendo válida
                state.active_signals[symbol]['timestamp'] = time.time()

    # --- MOTOR PnL Y LIMPIEZA ---
    to_delete = []
    for sym, sig in state.active_signals.items():
        if sym not in state.klines_5m: continue
        
        curr_p = state.klines_5m[sym]['close'].iloc[-1]
        entry = sig['entry']
        
        if sig['side'] == 'LONG':
            pnl_pct = ((curr_p - entry) / entry) * 100
        else:
            pnl_pct = ((entry - curr_p) / entry) * 100
            
        sig['pnl'] = pnl_pct

        # Stop Loss Dinámico o Cierre
        if pnl_pct <= -1.5:
            to_delete.append(sym)
        elif pnl_pct >= 0.2:
            sig['status'] = f"✅ +{pnl_pct:.2f}%"
        elif pnl_pct < 0:
            sig['status'] = f"⚠️ {pnl_pct:.2f}%"
        else:
            sig['status'] = f"🔄 +{pnl_pct:.2f}%"

    for sym in to_delete:
        del state.active_signals[sym]

# ==========================================
# WEBSOCKETS & BACKGROUND TASKS
# ==========================================
async def ws_multiplex_listener(session: aiohttp.ClientSession):
    while True:
        # Solo necesitamos el stream de 5 minutos, la lógica es limpia
        streams = "/".join([f"{sym.lower()}@kline_5m" for sym in state.pairs])
        ws_url = f"{BASE_WS_URL}{streams}"
        
        try:
            state.activity_msg = "[Conectando WebSockets Multiplexados (5m)...]"
            async with session.ws_connect(ws_url) as ws:
                state.ws_connection = ws  
                state.activity_msg = "[Procesando Flujo de Datos HFT (Ticks)]"
                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        data = payload['data']['k']
                        symbol = data['s']
                        close_price = float(data['c'])
                        high_price = float(data['h'])
                        low_price = float(data['l'])
                        is_closed = data['x']
                        
                        if symbol in state.klines_5m:
                            df = state.klines_5m[symbol]
                            df.iloc[-1, df.columns.get_loc('close')] = close_price
                            df.iloc[-1, df.columns.get_loc('high')] = high_price
                            df.iloc[-1, df.columns.get_loc('low')] = low_price
                            df.iloc[-1, df.columns.get_loc('open')] = float(data['o'])
                            df.iloc[-1, df.columns.get_loc('volume')] = float(data['v'])
                            
                            if is_closed:
                                new_row = df.iloc[-1].copy()
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).tail(500)
                                state.klines_5m[symbol] = df
                            
                            # --- INYECCIÓN HFT A MACRO ---
                            # Mantiene las velas de 15m y 1H actualizadas milisegundo a milisegundo 
                            # para que no te pierdas el cruce de bandas Macro
                            if symbol in state.klines_15m:
                                df_15 = state.klines_15m[symbol]
                                df_15.iloc[-1, df_15.columns.get_loc('close')] = close_price
                                if close_price > df_15.iloc[-1]['high']: df_15.iloc[-1, df_15.columns.get_loc('high')] = close_price
                                if close_price < df_15.iloc[-1]['low']: df_15.iloc[-1, df_15.columns.get_loc('low')] = close_price
                            
                            if symbol in state.klines_1h:
                                df_1h = state.klines_1h[symbol]
                                df_1h.iloc[-1, df_1h.columns.get_loc('close')] = close_price
                                if close_price > df_1h.iloc[-1]['high']: df_1h.iloc[-1, df_1h.columns.get_loc('high')] = close_price
                                if close_price < df_1h.iloc[-1]['low']: df_1h.iloc[-1, df_1h.columns.get_loc('low')] = close_price

                            asyncio.create_task(evaluate_logic(session, symbol))
                                
        except Exception as e:
            state.activity_msg = f"[Reconectando WebSocket...]"
            await asyncio.sleep(2)

async def periodic_rest_updater(session: aiohttp.ClientSession):
    while True:
        await asyncio.sleep(15) 
        state.activity_msg = "[Actualizando Estado Macro por REST...]"
        await sync_klines(session, '1h', state.klines_1h, state.pairs)
        await sync_klines(session, '15m', state.klines_15m, state.pairs)
        
        for sym in state.pairs:
            await evaluate_logic(session, sym)
        state.activity_msg = "[Procesando Flujo de Datos HFT (Ticks)]"

async def periodic_pairs_updater(session: aiohttp.ClientSession):
    while True:
        await asyncio.sleep(900) 
        old_pairs = set(state.pairs)
        await initialize_pairs(session)
        new_pairs = set(state.pairs)
        
        added_pairs = list(new_pairs - old_pairs)
        if added_pairs:
            state.activity_msg = f"[Nuevos pares detectados. Sincronizando...]"
            await sync_klines(session, '1h', state.klines_1h, added_pairs)
            await sync_klines(session, '15m', state.klines_15m, added_pairs)
            await sync_klines(session, '5m', state.klines_5m, added_pairs)
            if state.ws_connection and not state.ws_connection.closed:
                await state.ws_connection.close()

# ==========================================
# INTERFAZ GRÁFICA (RICH UI)
# ==========================================
def generate_ui() -> Layout:
    layout = Layout()
    layout.split(
        Layout(name="header", size=3),
        Layout(name="main")
    )
    layout["main"].split_row(
        Layout(name="radar", ratio=1),
        Layout(name="signals", ratio=1)
    )

    uptime = int(time.time() - state.start_time)
    header_text = Text()
    header_text.append(" [● ONLINE] ", style="bold green")
    header_text.append(f"| Ping: {state.ping}ms | Uptime: {uptime}s | ")
    header_text.append(state.activity_msg, style="bold cyan")
    layout["header"].update(Panel(header_text, title="HFT Crypto Scanner - Engine Status", style="bold blue", box=box.ROUNDED))

    radar_table = Table(expand=True, box=box.SIMPLE_HEAVY)
    radar_table.add_column("PAR", justify="left", style="cyan", no_wrap=True)
    radar_table.add_column("DIR", justify="center", style="bold")
    radar_table.add_column("RSI 1H", justify="right")
    radar_table.add_column("RSI 15m", justify="right")
    radar_table.add_column("RSI 5m", justify="right")
    radar_table.add_column("DIST. BANDA", justify="right")

    for sym, data in sorted(state.radar.items(), key=lambda x: x[1]['dist']):
        color = "green" if data['side'] == 'LONG' else "red"
        radar_table.add_row(
            sym, Text(data['side'], style=color),
            f"{data['rsi_1h']:.1f}", f"{data['rsi_15m']:.1f}",
            f"{data['rsi_5m']:.1f}", 
            f"{data['dist']:.2f}%"
        )
    layout["radar"].update(Panel(radar_table, title="[yellow]Radar de Aproximación (1H + 15m Alineados)", border_style="yellow"))

    signals_table = Table(expand=True, box=box.SIMPLE_HEAVY)
    signals_table.add_column("PAR", justify="left", style="white", no_wrap=True)
    signals_table.add_column("BB(5m) RAW", justify="right", style="gray70")
    signals_table.add_column("ENTRADA", justify="right", style="cyan")
    signals_table.add_column("MURO LIQ.", justify="right", style="magenta")
    signals_table.add_column("ESTADO / PnL", justify="center", style="bold") 

    current_t = time.time()
    # Limpiador de señales viejas (1 hora en pantalla)
    to_delete = [sym for sym, data in state.active_signals.items() if current_t - data['timestamp'] > 3600]
    for sym in to_delete:
        del state.active_signals[sym]

    for sym, data in state.active_signals.items():
        side_color = "bold green" if data['side'] == 'LONG' else "bold red"
        
        pnl = data.get('pnl', 0.0)
        
        if "GATILLO" in data['status']:
            status_style = "bold blink magenta"
        elif pnl >= 0.2:
            status_style = "bold green"
        elif pnl < 0:
            status_style = "bold red"
        else:
            status_style = "bold cyan"

        vol_formatted = f"${data['wall_vol']:,.0f}"
        decimals = state.tick_sizes.get(sym, {}).get('decimals', 4)
        bb_formatted = f"{data.get('bb_value', 0.0):.{decimals}f}"
        entry_formatted = f"{data.get('entry', 0.0):.{decimals}f}"

        signals_table.add_row(
            Text(sym, style=side_color),
            bb_formatted, 
            entry_formatted,
            vol_formatted,
            Text(data['status'], style=status_style)
        )
    layout["signals"].update(Panel(signals_table, title="[bold white]SEÑALES ACTIVAS (Gatillo Estricto: 1H + 15m + 5m)", border_style="bold green"))

    return layout

# ==========================================
# BOOTSTRAP Y LOOP PRINCIPAL
# ==========================================
async def main():
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        await initialize_pairs(session)
        await sync_klines(session, '1h', state.klines_1h, state.pairs)
        await sync_klines(session, '15m', state.klines_15m, state.pairs)
        await sync_klines(session, '5m', state.klines_5m, state.pairs)
        
        ws_task = asyncio.create_task(ws_multiplex_listener(session))
        rest_task = asyncio.create_task(periodic_rest_updater(session))
        pairs_updater_task = asyncio.create_task(periodic_pairs_updater(session))
        
        with Live(generate_ui(), refresh_per_second=4, screen=True) as live:
            while True:
                await asyncio.sleep(0.25)
                live.update(generate_ui())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Apagado del motor HFT completado.")
