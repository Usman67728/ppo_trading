import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os

# --- CONFIG ---
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
SYMBOL = "GBPJPY"                # main currency pair
TIMEFRAME = mt5.TIMEFRAME_M1     # 1-minute data
CHUNK_DAYS = 30                  # smaller chunks for 1-min data
OUTPUT_FILE = "GBPJPY_M1.csv"
LOCAL_TZ = "Asia/Karachi"        # your local timezone

# --- INIT MT5 ---
if not mt5.initialize(path=MT5_PATH):
    print("initialize() failed:", mt5.last_error())
    raise SystemExit


def ensure_symbol(symbol: str) -> bool:
    """Ensure symbol is available and visible in Market Watch."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.symbol_select(symbol, True)
    if not info.visible:
        return mt5.symbol_select(symbol, True)
    return True


def convert_to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MT5 UTC timestamps to local timezone."""
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["time"] = (
            df["time"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(LOCAL_TZ)
            .dt.tz_localize(None)
        )
    return df


def fetch_all_by_time_windows(symbol: str) -> pd.DataFrame | None:
    """Fetch full history in safe windows (avoid >99,999 bars per request)."""
    if not ensure_symbol(symbol):
        print(f"❌ {symbol}: not found/visible in Market Watch.")
        return None

    end = datetime.now()  # local time
    frames, total = [], 0

    while True:
        start = end - timedelta(days=CHUNK_DAYS)
        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)
        if rates is None or len(rates) == 0:
            break

        df = pd.DataFrame(rates)
        if "time" not in df.columns:
            end = start - timedelta(seconds=1)
            continue

        df = convert_to_local(df)
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)

        frames.append(df)
        total += len(df)
        print(f"🧩 {symbol}: batch={len(df)} total={total}")

        end = start - timedelta(seconds=1)
        if end < datetime.now() - timedelta(days=5 * 365):
            break

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset="time", inplace=True)
    out.sort_values("time", inplace=True)
    return out.reset_index(drop=True)


def update_dataset(symbol, existing_last_time=None):
    """Fetch only new data after existing_last_time."""
    if not ensure_symbol(symbol):
        print(f"❌ {symbol}: not found.")
        return None
    if existing_last_time is None:
        return fetch_all_by_time_windows(symbol)

    start = existing_last_time + pd.Timedelta(minutes=1)
    end = datetime.now()
    rates = mt5.copy_rates_range(symbol, TIMEFRAME, start, end)
    if rates is None or len(rates) == 0:
        print(f"⏹ No new data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df = convert_to_local(df)
    df = df[["time", "open", "high", "low", "close", "tick_volume"]]
    df.rename(columns={"tick_volume": "volume"}, inplace=True)

    print(f"➕ {symbol}: fetched {len(df)} new rows")
    return df


# --- MAIN WORKFLOW ---
if os.path.exists(OUTPUT_FILE):
    print("📂 Existing GBPJPY_M1.csv found. Updating...")
    df_all = pd.read_csv(OUTPUT_FILE, parse_dates=["time"])
    last_time = df_all["time"].max()

    new_data = update_dataset(SYMBOL, existing_last_time=last_time)
    if new_data is not None:
        df_all = pd.concat([df_all, new_data], ignore_index=True)
        df_all.drop_duplicates(subset="time", inplace=True)
        df_all.sort_values("time", inplace=True)
else:
    print("🆕 No dataset found. Fetching full history...")
    df_all = fetch_all_by_time_windows(SYMBOL)

# --- Add Volume Spike Feature ---
if df_all is not None and "volume" in df_all.columns:
    df_all["vol_spike"] = df_all["volume"] / df_all["volume"].rolling(window=20, min_periods=1).mean()

# --- Save ---
df_all.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Saved {OUTPUT_FILE} with {len(df_all)} rows (local time)")

mt5.shutdown()
