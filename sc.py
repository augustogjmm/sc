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
BASE_WS_URL = "wss://stream.binance.com:9443/ws" # Endpoint WS crudo para subscripciones
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
        self.last_eval = {} # Throttle para no ahogar el event loop
        
        self.ws_connection = None 
        self.activity_msg = "Iniciando sistema..."
        self.ping = 0
        self.start_time = time.time()

state = ScannerState()

# ==========================================
# MOTOR MATEMÁTICO (Vectorizado y Optimizado)
# ==========================================
def calculate_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI Vectorizado: Extremadamente rápido, no ahoga la CPU."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
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
    """Calcula indicadores. Retorna series completas limitadas a 150 para optimizar."""
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
    state.activity_msg = "[Recopilando Exchange Info y Top Pares por Volumen...]"
    
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

    # Utilizamos exclusivamente volumen Binance para evitar colisiones y límites de CoinGecko
    tickers = await fetch_json(session, f"{BASE_REST_URL}/api/v3/ticker/24hr")
    vol_pairs = [{'symbol': t['symbol'], 'vol': float(t['quoteVolume'])} for t in tickers if t['symbol'] in valid_binance_pairs]
    vol_pairs.sort(key=lambda x: x['vol'], reverse=True)
    
    state.pairs = [p['symbol'] for p in vol_pairs[:TOP_N_PAIRS]]
    state.tick_sizes = {sym: valid_binance_pairs[sym] for sym in state.pairs}

async def fetch_historical_klines(session: aiohttp.ClientSession, symbol: str, interval: str, limit: int = 150):
    # Límite reducido a 150: Suficiente para convergencia RSI, salva RAM y tiempo
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    data = await fetch_json(session, f"{BASE_REST_URL}/api/v3/klines", params)
    df = pd.DataFrame(data, columns=['open_time', 'open', 'high', 'low', 'close', 'volume', 
                                     'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
    # Convertimos tipos críticos para Pandas
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)
    return df

async def sync_klines(session: aiohttp.ClientSession, interval: str, target_dict: dict, pairs_list: list):
    state.activity_msg = f"[Sincronizando Velas Iniciales {interval}...]"
    chunk_size = 20
    for i in range(0, len(pairs_list), chunk_size):
        chunk = pairs_list[i:i+chunk_size]
        tasks = [fetch_historical_klines(session, sym, interval) for sym in chunk]
        results = await asyncio.gather(*tasks)
        for sym, df in zip(chunk, results):
            target_dict[sym] = df
        await asyncio.sleep(0.2) # Respetamos rate limit de Binance inicial

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
# LÓGICA CORE DIRECTA Y ESTRICTA
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
        
    rsi_1h_val = rsi_1h_s.iloc[-1]
    rsi_15m_val = rsi_15m_s.iloc[-1]
    rsi_5m_val = rsi_5m_s.iloc[-1]
    
    upper_5m_val = upper_5m_s.iloc[-1]
    lower_5m_val = lower_5m_s.iloc[-1]
    current_price = df_5m['close'].iloc[-1]

    # --- DIRECTIVAS MACRO (1H + 15m) CON MEMORIA CORTA ---
    long_1h = any((rsi_1h_s.iloc[-i] < 30 and df_1h['low'].iloc[-i] <= lower_1h_s.iloc[-i]) for i in range(1, 4) if len(df_1h) >= i)
    short_1h = any((rsi_1h_s.iloc[-i] > 70 and df_1h['high'].iloc[-i] >= upper_1h_s.iloc[-i]) for i in range(1, 4) if len(df_1h) >= i)
    
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
            if any((rsi_5m_s.iloc[-i] < 30 and df_5m['low'].iloc[-i] <= lower_5m_s.iloc[-i]) for i in range(1, 3) if len(df_5m) >= i):
                trigger = True
                ref_price = lower_5m_val
                bb_value_raw = lower_5m_val
                
        elif radar_info['side'] == 'SHORT':
            if any((rsi_5m_s.iloc[-i] > 70 and df_5m['high'].iloc[-i] >= upper_5m_s.iloc[-i]) for i in range(1, 3) if len(df_5m) >= i):
                trigger = True
                ref_price = upper_5m_val
                bb_value_raw = upper_5m_val

        if trigger:
            # Solo añadir si no existe o si ya estaba cerrado en histórico
            if symbol not in state.active_signals or state.active_signals[symbol].get('closed', False):
                entry_price, vol = await analyze_microstructure(session, symbol, radar_info['side'], ref_price)
                state.active_signals[symbol] = {
                    'side': radar_info['side'],
                    'bb_value': bb_value_raw, 
                    'entry': float(entry_price),
                    'wall_vol': vol,
                    'timestamp': time.time(),
                    'status': '¡GATILLO EN VIVO!',
                    'pnl': 0.0,
                    'closed': False
                }

    # --- MOTOR PnL SIMÉTRICO Y REAL (TAKE PROFIT + STOP LOSS) ---
    for sym, sig in list(state.active_signals.items()):
        if sig.get('closed', False):
            # Limpiar señales cerradas después de 30 segundos
            if time.time() - sig.get('close_time', 0) > 30:
                del state.active_signals[sym]
            continue

        if sym not in state.klines_5m: continue
        
        curr_p = state.klines_5m[sym]['close'].iloc[-1]
        entry = sig['entry']
        
        if sig['side'] == 'LONG':
            pnl_pct = ((curr_p - entry) / entry) * 100
        else:
            pnl_pct = ((entry - curr_p) / entry) * 100
            
        sig['pnl'] = pnl_pct

        # Reglas Claras de Salida
        if pnl_pct <= -1.5:
            sig['status'] = f"🛑 STOP LOSS [{pnl_pct:.2f}%]"
            sig['closed'] = True
            sig['close_time'] = time.time()
        elif pnl_pct >= 1.5:
            sig['status'] = f"🏆 TAKE PROFIT [{pnl_pct:.2f}%]"
            sig['closed'] = True
            sig['close_time'] = time.time()
        elif pnl_pct >= 0.2:
            sig['status'] = f"✅ +{pnl_pct:.2f}%"
        elif pnl_pct < 0:
            sig['status'] = f"⚠️ {pnl_pct:.2f}%"
        else:
            sig['status'] = f"🔄 +{pnl_pct:.2f}%"

# ==========================================
# WEBSOCKETS & BACKGROUND TASKS
# ==========================================
async def ws_multiplex_listener(session: aiohttp.ClientSession):
    """Suscripción nativa a WebSockets de Binance evitando Rate Limits REST."""
    while True:
        try:
            state.activity_msg = "[Conectando WebSockets Maestros...]"
            async with session.ws_connect(BASE_WS_URL) as ws:
                state.ws_connection = ws  
                state.activity_msg = "[Procesando Flujo de Datos HFT (Ticks)]"
                
                # Suscribimos 1h, 15m y 5m por chunks para no sobrecargar el payload
                streams = []
                for sym in state.pairs:
                    s = sym.lower()
                    streams.extend([f"{s}@kline_1h", f"{s}@kline_15m", f"{s}@kline_5m"])
                
                for i in range(0, len(streams), 50):
                    payload = {
                        "method": "SUBSCRIBE",
                        "params": streams[i:i+50],
                        "id": i
                    }
                    await ws.send_json(payload)
                    await asyncio.sleep(0.1)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        payload = json.loads(msg.data)
                        if 'k' not in payload: continue # Omitimos ACKs
                        
                        data = payload['k']
                        symbol = data['s']
                        interval = data['i']
                        kline_start = data['t']
                        close_price = float(data['c'])
                        high_price = float(data['h'])
                        low_price = float(data['l'])
                        
                        # Determinar qué DataFrame actualizar
                        if interval == '1h': target_dict = state.klines_1h
                        elif interval == '15m': target_dict = state.klines_15m
                        elif interval == '5m': target_dict = state.klines_5m
                        else: continue

                        if symbol in target_dict:
                            df = target_dict[symbol]
                            last_start = df['open_time'].iloc[-1]
                            
                            if kline_start == last_start:
                                # Actualizar vela en formación (Solución al repintado falso)
                                df.iloc[-1, df.columns.get_loc('close')] = close_price
                                df.iloc[-1, df.columns.get_loc('high')] = high_price
                                df.iloc[-1, df.columns.get_loc('low')] = low_price
                                df.iloc[-1, df.columns.get_loc('volume')] = float(data['v'])
                            elif kline_start > last_start:
                                # Crear nueva vela limpiamente
                                new_row = df.iloc[-1].copy()
                                new_row['open_time'] = kline_start
                                new_row['open'] = float(data['o'])
                                new_row['high'] = high_price
                                new_row['low'] = low_price
                                new_row['close'] = close_price
                                new_row['volume'] = float(data['v'])
                                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True).tail(150)
                                target_dict[symbol] = df
                            
                            # Throttle: Evaluar lógica máx 1 vez por segundo por par (Solución al ahogo de Event Loop)
                            if interval == '5m': # Evaluamos solo con los pulsos veloces de 5m
                                now = time.time()
                                if now - state.last_eval.get(symbol, 0) > 1.0:
                                    state.last_eval[symbol] = now
                                    asyncio.create_task(evaluate_logic(session, symbol))
                                
        except Exception as e:
            state.activity_msg = f"[Reconectando WebSocket: {str(e)}]"
            await asyncio.sleep(2)

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

    for sym, data in state.active_signals.items():
        side_color = "bold green" if data['side'] == 'LONG' else "bold red"
        
        if "TAKE PROFIT" in data['status']:
            status_style = "bold blink yellow"
        elif "STOP LOSS" in data['status']:
            status_style = "bold blink red"
        elif "GATILLO" in data['status']:
            status_style = "bold blink magenta"
        elif data.get('pnl', 0) >= 0.2:
            status_style = "bold green"
        elif data.get('pnl', 0) < 0:
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
    connector = aiohttp.TCPConnector(limit=50) # Límite sano de sockets
    async with aiohttp.ClientSession(connector=connector) as session:
        await initialize_pairs(session)
        await sync_klines(session, '1h', state.klines_1h, state.pairs)
        await sync_klines(session, '15m', state.klines_15m, state.pairs)
        await sync_klines(session, '5m', state.klines_5m, state.pairs)
        
        ws_task = asyncio.create_task(ws_multiplex_listener(session))
        
        with Live(generate_ui(), refresh_per_second=4, screen=True) as live:
            while True:
                await asyncio.sleep(0.25)
                live.update(generate_ui())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Apagado del motor HFT completado.")
