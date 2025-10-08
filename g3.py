import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
import os

# --- CONFIG ---
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"
MAIN_SYMBOL = "XAUUSD"
SUPPORTING = ["XAGUSD", "XPTUSD", "XPDUSD"]  # silver, platinum, palladium
TIMEFRAME = mt5.TIMEFRAME_M1  # 1-minute timeframe
CHUNK_DAYS = 30               # smaller chunks to avoid 99,999-bar limit
OUTPUT_FILE = "Gold_Metals_M1.csv"
LOCAL_TZ = pytz.timezone("Asia/Karachi")

# --- INIT MT5 ---
if not mt5.initialize(path=MT5_PATH):
    print("initialize() failed:", mt5.last_error())
    raise SystemExit


def ensure_symbol(symbol: str) -> bool:
    """Ensure symbol is visible in Market Watch."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return mt5.symbol_select(symbol, True)
    if not info.visible:
        return mt5.symbol_select(symbol, True)
    return True


def convert_to_local(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MT5 UTC timestamps to local timezone."""
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["time"] = df["time"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    return df


def fetch_all_by_time_windows(symbol: str, keep_ohlc: bool) -> pd.DataFrame | None:
    """Fetch full history in safe windows."""
    if not ensure_symbol(symbol):
        print(f"❌ {symbol}: not found/visible in Market Watch.")
        return None

    end_local = datetime.now(LOCAL_TZ)
    frames, total = [], 0

    while True:
        start_local = end_local - timedelta(days=CHUNK_DAYS)
        # Convert to UTC for MT5
        start_utc = start_local.astimezone(timezone.utc)
        end_utc = end_local.astimezone(timezone.utc)

        rates = mt5.copy_rates_range(symbol, TIMEFRAME, start_utc, end_utc)
        if rates is None or len(rates) == 0:
            break

        df = pd.DataFrame(rates)
        if "time" not in df.columns:
            end_local = start_local - timedelta(seconds=1)
            continue

        df = convert_to_local(df)

        if keep_ohlc:
            df = df[["time", "open", "high", "low", "close", "tick_volume"]]
            df.rename(columns={"tick_volume": "volume"}, inplace=True)
        else:
            df = df[["time", "high", "low", "close"]]
            df.rename(
                columns={
                    "high": f"{symbol}_high",
                    "low": f"{symbol}_low",
                    "close": f"{symbol}_close",
                },
                inplace=True,
            )

        frames.append(df)
        total += len(df)
        print(f"🧩 {symbol}: batch={len(df)} total={total}")

        end_local = start_local - timedelta(seconds=1)
        if end_local < datetime.now(LOCAL_TZ) - timedelta(days=5 * 365):
            break

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    out.drop_duplicates(subset="time", inplace=True)
    out.sort_values("time", inplace=True)
    return out.reset_index(drop=True)


def update_dataset(symbol, keep_ohlc, existing_last_time=None):
    """Fetch only new data after existing_last_time."""
    if not ensure_symbol(symbol):
        print(f"❌ {symbol}: not found.")
        return None
    if existing_last_time is None:
        return fetch_all_by_time_windows(symbol, keep_ohlc)

    start_local = existing_last_time + pd.Timedelta(minutes=1)
    end_local = datetime.now(LOCAL_TZ)
    start_utc = LOCAL_TZ.localize(start_local).astimezone(timezone.utc)
    end_utc = end_local.astimezone(timezone.utc)

    rates = mt5.copy_rates_range(symbol, TIMEFRAME, start_utc, end_utc)
    if rates is None or len(rates) == 0:
        print(f"⏹ No new data for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df = convert_to_local(df)

    if keep_ohlc:
        df = df[["time", "open", "high", "low", "close", "tick_volume"]]
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
    else:
        df = df[["time", "high", "low", "close"]]
        df.rename(
            columns={
                "high": f"{symbol}_high",
                "low": f"{symbol}_low",
                "close": f"{symbol}_close",
            },
            inplace=True,
        )

    print(f"➕ {symbol}: fetched {len(df)} new rows")
    return df


# --- MAIN WORKFLOW ---
if os.path.exists(OUTPUT_FILE):
    print("📂 Existing Gold_Metals_M1.csv found. Updating...")
    df_all = pd.read_csv(OUTPUT_FILE, parse_dates=["time"])
    last_time = df_all["time"].max()

    # Update gold (main)
    new_gold = update_dataset(MAIN_SYMBOL, keep_ohlc=True, existing_last_time=last_time)
    if new_gold is not None:
        df_all = pd.concat([df_all, new_gold], ignore_index=True)

    # Update supporting metals
    for sym in SUPPORTING:
        new_sym = update_dataset(sym, keep_ohlc=False, existing_last_time=last_time)
        if new_sym is not None:
            df_all = df_all.merge(new_sym, on="time", how="left")

    # Clean incomplete rows
    before = len(df_all)
    df_all.dropna(inplace=True)
    after = len(df_all)
    print(f"🧹 Dropped {before - after} rows with missing values. Final: {after}")

else:
    print("🆕 No dataset found. Fetching full history...")
    df_gold = fetch_all_by_time_windows(MAIN_SYMBOL, keep_ohlc=True)
    df_all = df_gold.copy()

    for sym in SUPPORTING:
        df_sym = fetch_all_by_time_windows(sym, keep_ohlc=False)
        if df_sym is not None:
            df_all = df_all.merge(df_sym, on="time", how="left")
        else:
            df_all[f"{sym}_high"] = pd.NA
            df_all[f"{sym}_low"] = pd.NA
            df_all[f"{sym}_close"] = pd.NA

    df_all.dropna(inplace=True)

# --- Add Volume Spike Feature ---
if "volume" in df_all.columns:
    df_all["vol_spike"] = df_all["volume"] / df_all["volume"].rolling(window=20, min_periods=1).mean()

# --- Save ---
df_all.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Saved {OUTPUT_FILE} with {len(df_all)} rows (local time)")

mt5.shutdown()
