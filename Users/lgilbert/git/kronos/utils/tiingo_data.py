from utils.tiingo_data import TiingoClient
from datetime import datetime, timezone
import pandas as pd
# Load secrets from .env (TIINGO_API_TOKEN) if python-dotenv is available.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

client = TiingoClient()  # reads TIINGO_API_TOKEN from the environment

end = datetime.now(timezone.utc)
start = end - pd.DateOffset(years=5)

df = client.download_klines(
    symbol='XAUUSD',
    frequency='1d',
    start=start,
    end=end,
)

print(f'Downloaded {len(df)} candles from {df.index[0].date()} to {df.index[-1].date()}')

# Format for kronos: timestamps,open,close,high,low,volume,amount
out = pd.DataFrame({
    'timestamps': df.index.strftime('%Y-%m-%d'),
    'open': df['open'],
    'close': df['close'],
    'high': df['high'],
    'low': df['low'],
    'volume': df['volume'],
    'amount': df['volume'],  # Tiingo forex has no amount; use volume as placeholder
})

out_path = 'data/XAUUSD_1d_5y.csv'
out.to_csv(out_path, index=False)
print(f'Saved to {out_path}')
print(out.head())
print('...')
print(out.tail())





