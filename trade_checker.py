"""
TRADE CHECKER - Pre-Trade Checklist Based on Data Analysis
===========================================================
Run this BEFORE entering any trade to check if it matches
your winning patterns or losing patterns.

Update the dictionaries below with your own trading data.
"""

from datetime import datetime
import sys

# =============================================================================
# TRADING RULES (Update these with your own data analysis results)
# =============================================================================

# Tickers to AVOID - update with your own blacklist based on historical P&L
# Format: 'TICKER': estimated_loss (negative number)
BLACKLIST_TICKERS = {
    # Example entries - replace with your actual data
    'EXAMPLE1': -5000,
    'EXAMPLE2': -3000,
}

# Tickers that WORK for you - update with your winners
# Format: 'TICKER': estimated_profit (positive number)
WHITELIST_TICKERS = {
    # Example entries - replace with your actual data
    'SPY': 1000,
    'AAPL': 500,
}

# Day of week patterns
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# P&L by day + option type - update with your own patterns
# Run your analysis to find which day/type combos work for you
DAY_OPTION_PNL = {
    ('Monday', 'PUT'): 1000,
    ('Monday', 'CALL'): -500,
    ('Tuesday', 'PUT'): -200,
    ('Tuesday', 'CALL'): -300,
    ('Wednesday', 'PUT'): 100,
    ('Wednesday', 'CALL'): -100,
    ('Thursday', 'PUT'): 50,
    ('Thursday', 'CALL'): -400,
    ('Friday', 'PUT'): 200,
    ('Friday', 'CALL'): -1000,
}


def get_trade_score(ticker, option_type, trade_date=None, last_trade_was_loss=False):
    """
    Score a potential trade from -100 (terrible) to +100 (great).
    Returns (score, warnings, positives)
    """
    score = 0
    warnings = []
    positives = []

    ticker = ticker.upper()
    option_type = option_type.upper()

    if trade_date is None:
        trade_date = datetime.now()

    day_name = DAY_NAMES[trade_date.weekday()]
    day_of_month = trade_date.day

    # =================================================================
    # CHECK 1: Blacklisted ticker
    # =================================================================
    if ticker in BLACKLIST_TICKERS:
        score -= 40
        warnings.append(f"BLACKLIST: {ticker} is on your blacklist")

    # =================================================================
    # CHECK 2: Whitelisted ticker
    # =================================================================
    if ticker in WHITELIST_TICKERS:
        score += 25
        positives.append(f"PROVEN WINNER: {ticker} is on your whitelist")

    # =================================================================
    # CHECK 3: Day + Option Type combo
    # =================================================================
    combo = (day_name, option_type)
    if combo in DAY_OPTION_PNL:
        pnl = DAY_OPTION_PNL[combo]
        if pnl > 500:
            score += 30
            positives.append(f"GREAT COMBO: {day_name} {option_type}s are historically profitable")
        elif pnl > 0:
            score += 10
            positives.append(f"OK COMBO: {day_name} {option_type}s are slightly positive")
        elif pnl > -500:
            score -= 15
            warnings.append(f"WEAK COMBO: {day_name} {option_type}s are slightly negative")
        else:
            score -= 35
            warnings.append(f"TERRIBLE COMBO: {day_name} {option_type}s are historically losing")

    # =================================================================
    # CHECK 4: Time of month patterns
    # =================================================================
    if 1 <= day_of_month <= 10:
        # Update based on your own early-month patterns
        score -= 10
        warnings.append(f"EARLY MONTH ({day_of_month}th): Check your early-month stats")
    elif 11 <= day_of_month <= 20:
        score += 10
        positives.append(f"MID MONTH ({day_of_month}th): Often a good trading period")
    else:
        score -= 5
        warnings.append(f"LATE MONTH ({day_of_month}th): Check your late-month stats")

    # =================================================================
    # CHECK 5: Coming off a loss (tilt check)
    # =================================================================
    if last_trade_was_loss:
        score -= 25
        warnings.append("TILT WARNING: You're coming off a loss - win rate drops significantly")
        warnings.append("RECOMMENDATION: Skip this trade or reduce size by 50%+")

    # =================================================================
    # CHECK 6: Hold time recommendation
    # =================================================================
    positives.append("REMINDER: Longer holds historically have better win rates")

    return score, warnings, positives


def print_trade_analysis(ticker, option_type, trade_date=None, last_trade_was_loss=False):
    """Pretty print the trade analysis"""
    score, warnings, positives = get_trade_score(ticker, option_type, trade_date, last_trade_was_loss)

    print("\n" + "="*60)
    print(f"TRADE CHECK: {ticker} {option_type}")
    print("="*60)

    if trade_date:
        day_name = DAY_NAMES[trade_date.weekday()]
        print(f"Date: {trade_date.strftime('%Y-%m-%d')} ({day_name})")
    else:
        print(f"Date: Today ({DAY_NAMES[datetime.now().weekday()]})")

    print(f"\nOVERALL SCORE: {score}")

    if score >= 30:
        print("VERDICT: LOOKS GOOD - Trade aligns with your winning patterns")
    elif score >= 0:
        print("VERDICT: MARGINAL - Proceed with caution, reduce size")
    elif score >= -30:
        print("VERDICT: WEAK - Consider skipping or minimal size only")
    else:
        print("VERDICT: AVOID - This trade matches your losing patterns")

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if positives:
        print(f"\nPOSITIVES ({len(positives)}):")
        for p in positives:
            print(f"  + {p}")

    print("\n" + "="*60)

    return score


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python trade_checker.py TICKER CALL/PUT [loss]")
        print("Example: python trade_checker.py NVDA PUT")
        print("Example: python trade_checker.py AAPL CALL loss  (if last trade was a loss)")
        sys.exit(1)

    ticker = sys.argv[1]
    option_type = sys.argv[2]
    last_was_loss = len(sys.argv) > 3 and sys.argv[3].lower() == 'loss'

    print_trade_analysis(ticker, option_type, datetime.now(), last_was_loss)
