# SC — Scanner experimental de mercado

Proyecto personal en Python de experimentación con **datos públicos de Binance, análisis técnico y procesamiento asíncrono**.

El código principal, `sc.py`, utiliza peticiones HTTP asíncronas, procesamiento con pandas y NumPy, y una interfaz en terminal basada en Rich.

## Exploración técnica

- Descubrimiento de pares spot USDT desde el servicio de Binance.
- Consulta de velas y procesamiento de indicadores como RSI y bandas de Bollinger.
- Manejo de estado en memoria para señales y datos de mercado.
- Experimentación con una interfaz de terminal.

## Ejecución experimental

Se recomienda Python 3.10 o superior. Instalar en un entorno virtual `aiohttp`, `pandas`, `numpy` y `rich` según los imports observados en el código. Ejecutar:

```bash
python sc.py
```

**Estado:** prototipo de investigación, no producto de producción. No se publican aquí pruebas automatizadas que certifiquen comportamiento, rendimiento o disponibilidad. Los endpoints y restricciones del proveedor pueden cambiar.

No constituye asesoramiento financiero ni demuestra un resultado operativo rentable.

**Autor:** [Augusto García Juárez Matorras](https://www.linkedin.com/in/augustog/).
