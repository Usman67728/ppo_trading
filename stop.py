import MetaTrader5 as mt5

def connect_mt5():
    """Initialize MT5 connection."""
    if not mt5.initialize():
        raise RuntimeError(f"Failed to initialize MT5: {mt5.last_error()}")
    print("[INFO] Connected to MetaTrader 5")

def close_all_trades():
    """Close all open trades for all symbols."""
    positions = mt5.positions_get()

    if positions is None:
        print("[INFO] No open positions found.")
        return
    elif len(positions) == 0:
        print("[INFO] No trades to close.")
        return

    print(f"[INFO] Found {len(positions)} open trades. Closing now...")

    for pos in positions:
        ticket = pos.ticket
        symbol = pos.symbol
        volume = pos.volume
        position_type = pos.type

        # Determine opposite order type
        if position_type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask

        close_request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": 20251008,
            "comment": "Python close all trades",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        print(f"[CLOSING] {symbol} | Ticket: {ticket} | Volume: {volume} | Type: {'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'}")
        result = mt5.order_send(close_request)

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[SUCCESS] Closed {symbol} | Ticket: {ticket}")
        else:
            print(f"[FAILED] Could not close {symbol} | Ticket: {ticket} | Retcode: {result.retcode}")

def shutdown_mt5():
    mt5.shutdown()
    print("[INFO] MT5 connection closed.")

if __name__ == "__main__":
    try:
        connect_mt5()
        close_all_trades()
    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        shutdown_mt5()
