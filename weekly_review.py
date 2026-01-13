"""
WEEKLY TRADING REVIEW
=====================
Run this every weekend to review your week and plan the next.

Usage:
    python weekly_review.py              # Review last 7 days
    python weekly_review.py 2026-01-06   # Review week containing that date
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES, POSITIONS_FILE
except ImportError:
    # Update config.py with your actual paths (see config_example.py)
    TRADE_FILES = ["./data/trades.csv"]
    POSITIONS_FILE = "./data/positions.csv"

# Your proven patterns (from analysis)
BLACKLIST_TICKERS = ['MSTR', 'AMZN', 'SMCI', 'SBET', 'GOOG', 'GME', 'IBIT']
WHITELIST_TICKERS = ['ETHA', 'NVDA', 'AAPL', 'SPY', 'KMX', 'INTC', 'TSLA', 'MSFT']

# =============================================================================
# LOAD ALL HISTORICAL DATA
# =============================================================================

def load_all_data():
    """Load and process all trade data."""
    all_trades = []
    for f in TRADE_FILES:
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, encoding='utf-8-sig')
                df = df[df['Run Date'].notna()]
                df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage', case=False, na=False)]
                all_trades.append(df)
            except:
                pass

    if not all_trades:
        return pd.DataFrame()

    trades_df = pd.concat(all_trades, ignore_index=True)

    # Filter to actual trades
    trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
    exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
    trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)]
    trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

    # Parse details
    def parse_trade(row):
        action = str(row['Action'])
        symbol = str(row['Symbol'])

        if 'BOUGHT OPENING' in action: direction = 'BUY_OPEN'
        elif 'SOLD CLOSING' in action: direction = 'SELL_CLOSE'
        elif 'SOLD OPENING' in action: direction = 'SELL_OPEN'
        elif 'BOUGHT CLOSING' in action: direction = 'BUY_CLOSE'
        elif 'ASSIGNED' in action: direction = 'ASSIGNED'
        elif 'EXPIRED' in action: direction = 'EXPIRED'
        elif 'YOU BOUGHT' in action: direction = 'BUY'
        elif 'YOU SOLD' in action: direction = 'SELL'
        else: direction = 'OTHER'

        is_option = 'CALL' in action or 'PUT' in action
        option_type = 'CALL' if 'CALL' in action else ('PUT' if 'PUT' in action else None)

        underlying = symbol
        if is_option:
            import re
            match = re.search(r'\(([A-Z]+)\)', action)
            if match: underlying = match.group(1)

        return pd.Series({
            'direction': direction,
            'is_option': is_option,
            'option_type': option_type,
            'underlying': underlying
        })

    parsed = trades_df.apply(parse_trade, axis=1)
    trades_df = pd.concat([trades_df, parsed], axis=1)
    trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
    trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
    trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')
    trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')

    # Account type
    def categorize_account(account):
        account = str(account).lower()
        if '401k' in account: return '401k'
        elif 'roth' in account: return 'Roth IRA'
        elif 'traditional' in account: return 'Traditional IRA'
        elif 'individual' in account: return 'Individual'
        elif 'option' in account: return 'Options'
        else: return 'Other'

    trades_df['account_type'] = trades_df['Account'].apply(categorize_account)
    trades_df['day_of_week'] = trades_df['Run Date'].dt.dayofweek
    trades_df['day_name'] = trades_df['Run Date'].dt.day_name()

    return trades_df


def get_week_range(target_date=None):
    """Get the Monday-Friday range for the week containing target_date."""
    if target_date is None:
        target_date = datetime.now()

    # Find Monday of that week
    monday = target_date - timedelta(days=target_date.weekday())
    friday = monday + timedelta(days=4)

    return monday.replace(hour=0, minute=0, second=0, microsecond=0), \
           friday.replace(hour=23, minute=59, second=59, microsecond=999999)


# =============================================================================
# WEEKLY REVIEW
# =============================================================================

def weekly_review(trades_df, week_start, week_end):
    """Generate the weekly review."""

    # Filter to this week
    week_trades = trades_df[
        (trades_df['Run Date'] >= week_start) &
        (trades_df['Run Date'] <= week_end)
    ].copy()

    # Get historical data (everything before this week)
    historical = trades_df[trades_df['Run Date'] < week_start].copy()

    print("\n" + "="*70)
    print(f"WEEKLY TRADING REVIEW")
    print(f"Week of {week_start.strftime('%B %d, %Y')} - {week_end.strftime('%B %d, %Y')}")
    print("="*70)

    if len(week_trades) == 0:
        print("\nNo trades found for this week.")
        return

    # ==========================================================================
    # 1. WEEK SUMMARY
    # ==========================================================================
    print("\n" + "-"*70)
    print("1. WEEK SUMMARY")
    print("-"*70)

    # Calculate P&L from transactions
    week_pnl = week_trades['Amount'].sum()

    # Count trades (opening transactions)
    opens = week_trades[week_trades['direction'].isin(['BUY_OPEN', 'SELL_OPEN', 'BUY', 'SELL'])]
    closes = week_trades[week_trades['direction'].isin(['SELL_CLOSE', 'BUY_CLOSE', 'ASSIGNED', 'EXPIRED'])]

    # Wins vs losses from closing transactions
    winning_closes = closes[closes['Amount'] > 0]
    losing_closes = closes[closes['Amount'] < 0]

    total_closes = len(closes)
    wins = len(winning_closes)
    losses = len(losing_closes)
    win_rate = wins / total_closes if total_closes > 0 else 0

    status = "PROFITABLE" if week_pnl > 0 else "LOSING"

    print(f"""
    Total Transactions:    {len(week_trades)}
    New Positions Opened:  {len(opens)}
    Positions Closed:      {len(closes)}

    Winning Closes:        {wins}
    Losing Closes:         {losses}
    Win Rate:              {win_rate:.0%}

    Week P&L:              ${week_pnl:,.2f}
    Status:                {status} WEEK
    """)

    # ==========================================================================
    # 2. DAILY BREAKDOWN
    # ==========================================================================
    print("-"*70)
    print("2. DAILY BREAKDOWN")
    print("-"*70)

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

    print(f"\n{'Day':<12} {'P&L':>12} {'Trades':>8} {'Opens':>8} {'Closes':>8}")
    print("-"*50)

    for i, day in enumerate(day_names):
        day_data = week_trades[week_trades['day_of_week'] == i]
        if len(day_data) > 0:
            day_pnl = day_data['Amount'].sum()
            day_opens = len(day_data[day_data['direction'].isin(['BUY_OPEN', 'SELL_OPEN', 'BUY', 'SELL'])])
            day_closes = len(day_data[day_data['direction'].isin(['SELL_CLOSE', 'BUY_CLOSE', 'ASSIGNED', 'EXPIRED'])])
            status = "+" if day_pnl > 0 else ""
            print(f"{day:<12} {status}${day_pnl:>10,.0f} {len(day_data):>8} {day_opens:>8} {day_closes:>8}")
        else:
            print(f"{day:<12} {'--':>12} {'0':>8} {'0':>8} {'0':>8}")

    # ==========================================================================
    # 3. BY TICKER
    # ==========================================================================
    print("\n" + "-"*70)
    print("3. TICKERS TRADED THIS WEEK")
    print("-"*70)

    ticker_summary = week_trades.groupby('underlying').agg({
        'Amount': 'sum',
        'direction': 'count'
    }).reset_index()
    ticker_summary.columns = ['ticker', 'pnl', 'transactions']
    ticker_summary = ticker_summary.sort_values('pnl', ascending=False)

    print(f"\n{'Ticker':<10} {'P&L':>12} {'Txns':>8} {'Status':<15}")
    print("-"*50)

    for _, row in ticker_summary.iterrows():
        status = ""
        if row['ticker'] in BLACKLIST_TICKERS:
            status = "!! BLACKLIST"
        elif row['ticker'] in WHITELIST_TICKERS:
            status = "OK - Whitelist"

        pnl_str = f"+${row['pnl']:,.0f}" if row['pnl'] > 0 else f"${row['pnl']:,.0f}"
        print(f"{row['ticker']:<10} {pnl_str:>12} {int(row['transactions']):>8} {status:<15}")

    # ==========================================================================
    # 4. CALLS VS PUTS
    # ==========================================================================
    print("\n" + "-"*70)
    print("4. CALLS VS PUTS THIS WEEK")
    print("-"*70)

    calls = week_trades[week_trades['option_type'] == 'CALL']
    puts = week_trades[week_trades['option_type'] == 'PUT']

    call_pnl = calls['Amount'].sum() if len(calls) > 0 else 0
    put_pnl = puts['Amount'].sum() if len(puts) > 0 else 0

    print(f"""
    CALLS: ${call_pnl:,.0f} ({len(calls)} transactions)
    PUTS:  ${put_pnl:,.0f} ({len(puts)} transactions)

    Historical reminder: Your PUTs have 62% win rate, CALLs have 45%
    """)

    # ==========================================================================
    # 5. BY ACCOUNT
    # ==========================================================================
    print("-"*70)
    print("5. BY ACCOUNT THIS WEEK")
    print("-"*70)

    account_summary = week_trades.groupby('account_type').agg({
        'Amount': 'sum',
        'direction': 'count'
    }).reset_index()
    account_summary.columns = ['account', 'pnl', 'transactions']
    account_summary = account_summary.sort_values('pnl', ascending=False)

    print(f"\n{'Account':<18} {'P&L':>12} {'Txns':>8}")
    print("-"*40)

    for _, row in account_summary.iterrows():
        pnl_str = f"+${row['pnl']:,.0f}" if row['pnl'] > 0 else f"${row['pnl']:,.0f}"
        print(f"{row['account']:<18} {pnl_str:>12} {int(row['transactions']):>8}")

    # ==========================================================================
    # 6. RULE VIOLATIONS
    # ==========================================================================
    print("\n" + "-"*70)
    print("6. RULE VIOLATIONS THIS WEEK")
    print("-"*70)

    violations = []

    # Check for Friday calls
    friday_calls = week_trades[
        (week_trades['day_of_week'] == 4) &
        (week_trades['option_type'] == 'CALL') &
        (week_trades['direction'] == 'BUY_OPEN')
    ]
    if len(friday_calls) > 0:
        fri_call_pnl = friday_calls['Amount'].sum()
        violations.append(f"Friday CALLs: {len(friday_calls)} opened (your worst pattern)")

    # Check for blacklist tickers
    blacklist_trades = week_trades[week_trades['underlying'].isin(BLACKLIST_TICKERS)]
    if len(blacklist_trades) > 0:
        bl_tickers = blacklist_trades['underlying'].unique().tolist()
        violations.append(f"Blacklist tickers traded: {', '.join(bl_tickers)}")

    # Check for early month trading (days 1-10)
    early_month = week_trades[week_trades['Run Date'].dt.day <= 10]
    if len(early_month) > len(week_trades) * 0.5:
        violations.append(f"Heavy early-month trading ({len(early_month)} trades) - historically your worst period")

    # Check for potential tilt (multiple losses in a row on same day)
    for day in week_trades['Run Date'].dt.date.unique():
        day_trades = week_trades[week_trades['Run Date'].dt.date == day].sort_values('Run Date')
        day_closes = day_trades[day_trades['direction'].isin(['SELL_CLOSE', 'BUY_CLOSE'])]
        if len(day_closes) >= 3:
            losses = day_closes[day_closes['Amount'] < 0]
            if len(losses) >= 3:
                violations.append(f"Potential tilt on {day}: {len(losses)} losing closes in one day")

    if violations:
        print("\n!! VIOLATIONS DETECTED:")
        for v in violations:
            print(f"   - {v}")
    else:
        print("\n   No major rule violations detected this week.")

    # ==========================================================================
    # 7. WINS & LOSSES DETAIL
    # ==========================================================================
    print("\n" + "-"*70)
    print("7. INDIVIDUAL TRADES")
    print("-"*70)

    # Get closing transactions with P&L
    closes_detail = closes[['Run Date', 'underlying', 'option_type', 'Amount', 'account_type']].copy()
    closes_detail = closes_detail.sort_values('Amount', ascending=False)

    print("\nBest trades this week:")
    for _, row in closes_detail.head(5).iterrows():
        if row['Amount'] > 0:
            print(f"   +${row['Amount']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'} | {row['account_type']}")

    print("\nWorst trades this week:")
    for _, row in closes_detail.tail(5).iloc[::-1].iterrows():
        if row['Amount'] < 0:
            print(f"   ${row['Amount']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'} | {row['account_type']}")

    # ==========================================================================
    # 8. COMPARISON TO HISTORICAL
    # ==========================================================================
    print("\n" + "-"*70)
    print("8. COMPARED TO YOUR HISTORY")
    print("-"*70)

    # Historical weekly stats
    if len(historical) > 0:
        hist_weekly = historical.groupby(historical['Run Date'].dt.to_period('W'))['Amount'].sum()
        avg_week = hist_weekly.mean()
        best_week = hist_weekly.max()
        worst_week = hist_weekly.min()

        print(f"""
    Your average week:     ${avg_week:,.0f}
    Your best week:        ${best_week:,.0f}
    Your worst week:       ${worst_week:,.0f}

    THIS WEEK:             ${week_pnl:,.0f}
    """)

        if week_pnl > avg_week:
            print("    This week: ABOVE AVERAGE")
        else:
            print("    This week: BELOW AVERAGE")

    # ==========================================================================
    # 9. OPEN POSITIONS CHECK
    # ==========================================================================
    print("\n" + "-"*70)
    print("9. POSITIONS OPENED THIS WEEK (Still Open?)")
    print("-"*70)

    # Show positions opened but not yet closed
    week_opens = week_trades[week_trades['direction'].isin(['BUY_OPEN', 'SELL_OPEN'])]
    week_close_symbols = week_trades[week_trades['direction'].isin(['SELL_CLOSE', 'BUY_CLOSE'])]['Symbol'].unique()

    still_open = week_opens[~week_opens['Symbol'].isin(week_close_symbols)]

    if len(still_open) > 0:
        print("\nPositions opened this week (may still be open):")
        for _, row in still_open.iterrows():
            print(f"   {row['underlying']} {row['option_type']} | Opened {row['Run Date'].strftime('%a %m/%d')} | {row['account_type']}")
    else:
        print("\n   All positions opened this week appear to be closed.")

    # ==========================================================================
    # 10. NEXT WEEK PREP
    # ==========================================================================
    print("\n" + "-"*70)
    print("10. NEXT WEEK PREPARATION")
    print("-"*70)

    next_monday = week_end + timedelta(days=3)
    next_week_day = next_monday.day

    month_period = "EARLY" if next_week_day <= 10 else ("MID" if next_week_day <= 20 else "LATE")

    print(f"""
    Next week starts: {next_monday.strftime('%B %d, %Y')}
    Month period: {month_period} MONTH
    """)

    if month_period == "EARLY":
        print("    !! WARNING: Early month is historically your worst period")
        print("    RECOMMENDATION: Reduce position sizes or trade less")
    elif month_period == "MID":
        print("    GOOD: Mid-month is historically your best period")
        print("    RECOMMENDATION: Normal trading, stick to your rules")
    else:
        print("    CAUTION: Late month is historically negative")
        print("    RECOMMENDATION: Be selective with trades")

    print(f"""
    REMINDERS FOR NEXT WEEK:
    - Monday PUTs are historically your best pattern
    - Avoid Friday CALLs (historically worst pattern)
    - Stick to whitelist: {', '.join(WHITELIST_TICKERS[:5])}...
    - Avoid blacklist: {', '.join(BLACKLIST_TICKERS[:5])}...
    - After any loss, reduce size or stop for the day
    - Smaller positions have higher win rates
    """)

    # ==========================================================================
    # 11. WEEKLY GRADE
    # ==========================================================================
    print("\n" + "="*70)
    print("WEEKLY GRADE")
    print("="*70)

    grade_score = 0

    # P&L contribution
    if week_pnl > 0:
        grade_score += 30
    elif week_pnl > -1000:
        grade_score += 15
    elif week_pnl > -5000:
        grade_score += 0
    else:
        grade_score -= 20

    # Win rate contribution
    if win_rate >= 0.6:
        grade_score += 25
    elif win_rate >= 0.5:
        grade_score += 15
    elif win_rate >= 0.4:
        grade_score += 5
    else:
        grade_score -= 10

    # Rule following
    grade_score -= len(violations) * 10

    # Whitelist/blacklist adherence
    whitelist_trades = len(week_trades[week_trades['underlying'].isin(WHITELIST_TICKERS)])
    blacklist_trades_count = len(week_trades[week_trades['underlying'].isin(BLACKLIST_TICKERS)])

    if whitelist_trades > blacklist_trades_count * 2:
        grade_score += 15
    elif blacklist_trades_count > whitelist_trades:
        grade_score -= 15

    # Determine grade
    if grade_score >= 50:
        grade = "A"
        comment = "Excellent week! You followed your rules and it paid off."
    elif grade_score >= 35:
        grade = "B"
        comment = "Good week. Minor improvements possible."
    elif grade_score >= 20:
        grade = "C"
        comment = "Average week. Review your rule violations."
    elif grade_score >= 5:
        grade = "D"
        comment = "Below average. Too many rule violations or losses."
    else:
        grade = "F"
        comment = "Poor week. Major rule violations or significant losses."

    print(f"""
    GRADE: {grade}

    {comment}

    Breakdown:
    - P&L: ${week_pnl:,.0f}
    - Win Rate: {win_rate:.0%}
    - Rule Violations: {len(violations)}
    - Whitelist Trades: {whitelist_trades}
    - Blacklist Trades: {blacklist_trades_count}
    """)

    print("="*70)
    print("END OF WEEKLY REVIEW")
    print("="*70)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Loading trade data...")
    trades_df = load_all_data()

    if len(trades_df) == 0:
        print("No trade data found!")
        sys.exit(1)

    # Determine which week to review
    if len(sys.argv) > 1:
        try:
            target_date = datetime.strptime(sys.argv[1], '%Y-%m-%d')
        except:
            print(f"Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        # Default to last week (most recent completed week)
        today = datetime.now()
        # Go back to find the most recent Friday
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0 and today.hour < 18:
            days_since_friday = 7  # If it's Friday before market close, review previous week
        last_friday = today - timedelta(days=days_since_friday)
        target_date = last_friday

    week_start, week_end = get_week_range(target_date)

    weekly_review(trades_df, week_start, week_end)
