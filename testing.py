import MetaTrader5 as mt5
from datetime import datetime

# --------------------------
# Custom Error Classes
# --------------------------
class MT5ConnectionError(Exception):
    pass

class SymbolError(Exception):
    pass

class OrderSendError(Exception):
    pass


# --------------------------
# Helper Functions
# --------------------------
def connect_mt5():
    """Initialize MetaTrader 5 connection."""
    if not mt5.initialize():
        raise MT5ConnectionError(f"MT5 initialization failed. Error: {mt5.last_error()}")
    print("[INFO] Connected to MetaTrader 5")

def get_account_info():
    """Fetch account info and confirm leverage."""
    account = mt5.account_info()
    if account is None:
        raise MT5ConnectionError("Unable to retrieve account info.")
    print(f"[INFO] Account: {account.login}, Leverage: {account.leverage}, Balance: {account.balance}")
    return account

def ensure_symbol(symbol):
    """Ensure symbol is available in Market Watch."""
    if not mt5.symbol_select(symbol, True):
        raise SymbolError(f"Failed to select symbol: {symbol}")
    print(f"[INFO] Symbol {symbol} is ready for trading")

def place_buy_order(symbol="XAUUSD", lot=0.35, sl_distance=2.0, tp_distance=4.0, deviation=50):
    """Place a market Buy order on given symbol."""
    
    # Get latest symbol info
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise SymbolError(f"Could not retrieve tick data for {symbol}")
    
    ask = tick.ask
    sl = ask - sl_distance
    tp = ask + tp_distance

    # Prepare order dictionary
    order_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY,
        "price": ask,
        "sl": sl,
        "tp": tp,
        "deviation": deviation,
        "magic": 20251008,
        "comment": f"Python auto-trade on {symbol}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    print(f"[INFO] Sending BUY order for {symbol} | Lot: {lot} | Price: {ask} | SL: {sl} | TP: {tp}")
    
    result = mt5.order_send(order_request)
    if result is None:
        raise OrderSendError("order_send() returned None")

    print(f"[RESULT] Retcode: {result.retcode}")
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        raise OrderSendError(f"Order failed. Retcode: {result.retcode}, Details: {result}")
    
    print(f"[SUCCESS] Order placed! Ticket: {result.order}")
    return result


def shutdown_mt5():
    """Close MT5 connection safely."""
    mt5.shutdown()
    print("[INFO] MT5 connection closed.")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    try:
        connect_mt5()
        get_account_info()
        ensure_symbol("XAUUSD")
        place_buy_order(symbol="XAUUSD", lot=0.35)
    except (MT5ConnectionError, SymbolError, OrderSendError) as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[UNEXPECTED ERROR] {e}")
    finally:
        shutdown_mt5()
