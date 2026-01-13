"""
DEEP TRADE ANALYSIS - Comprehensive Analytics
==============================================
Digging into every pattern in your trading data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES
    CSV_PATH = TRADE_FILES[0] if TRADE_FILES else "./data/trades.csv"
except ImportError:
    CSV_PATH = "./data/trades.csv"

print("Loading and preparing data...")
df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

# Filter to trades
trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
exclude_keywords = ['DIVIDEND', 'REINVESTMENT']

trades_df = df[df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)].copy()
trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

# Parse trade details
def parse_trade(row):
    action = row['Action']
    symbol = row['Symbol']

    if 'BOUGHT OPENING' in action:
        direction = 'BUY_OPEN'
    elif 'SOLD CLOSING' in action:
        direction = 'SELL_CLOSE'
    elif 'SOLD OPENING' in action:
        direction = 'SELL_OPEN'
    elif 'BOUGHT CLOSING' in action:
        direction = 'BUY_CLOSE'
    elif 'ASSIGNED' in action:
        direction = 'ASSIGNED'
    elif 'EXPIRED' in action:
        direction = 'EXPIRED'
    elif 'YOU BOUGHT' in action:
        direction = 'BUY'
    elif 'YOU SOLD' in action:
        direction = 'SELL'
    else:
        direction = 'OTHER'

    is_option = 'CALL' in action or 'PUT' in action or symbol.startswith('-')
    option_type = None
    if 'CALL' in action:
        option_type = 'CALL'
    elif 'PUT' in action:
        option_type = 'PUT'

    underlying = symbol
    if is_option and '(' in action:
        import re
        match = re.search(r'\(([A-Z]+)\)', action)
        if match:
            underlying = match.group(1)

    return pd.Series({
        'direction': direction,
        'is_option': is_option,
        'option_type': option_type,
        'underlying': underlying
    })

parsed = trades_df.apply(parse_trade, axis=1)
trades_df = pd.concat([trades_df, parsed], axis=1)
trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'])
trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')
trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')

# Account categorization
def categorize_account(account):
    if '401k' in account.lower():
        return '401k'
    elif 'roth' in account.lower():
        return 'Roth'
    elif 'traditional' in account.lower():
        return 'Traditional'
    elif 'individual' in account.lower():
        return 'Individual'
    elif 'option' in account.lower():
        return 'Options'
    else:
        return 'Other'

trades_df['account_type'] = trades_df['Account'].apply(categorize_account)

# Build closed positions
symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type']).agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'Run Date': ['min', 'max'],
    'direction': 'count',
    'Price': 'mean'
}).reset_index()

symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type',
                       'total_pnl', 'net_quantity', 'first_date', 'last_date',
                       'num_transactions', 'avg_price']

closed_positions = symbol_pnl[
    (abs(symbol_pnl['net_quantity']) <= 1) &
    (symbol_pnl['num_transactions'] >= 2)
].copy()

closed_positions['is_winner'] = closed_positions['total_pnl'] > 0
closed_positions['first_date'] = pd.to_datetime(closed_positions['first_date'])
closed_positions['last_date'] = pd.to_datetime(closed_positions['last_date'])
closed_positions['hold_days'] = (closed_positions['last_date'] - closed_positions['first_date']).dt.days
closed_positions['day_of_week'] = closed_positions['first_date'].dt.dayofweek
closed_positions['month'] = closed_positions['first_date'].dt.month
closed_positions['week_of_year'] = closed_positions['first_date'].dt.isocalendar().week
closed_positions['account_type'] = closed_positions['Account'].apply(categorize_account)

print(f"Loaded {len(closed_positions)} closed positions")
print("="*70)


# =============================================================================
# 1. CUMULATIVE P&L OVER TIME
# =============================================================================
print("\n" + "="*70)
print("1. CUMULATIVE P&L OVER TIME")
print("="*70)

# Sort by date and calculate running P&L
time_series = closed_positions.sort_values('last_date').copy()
time_series['cumulative_pnl'] = time_series['total_pnl'].cumsum()
time_series['trade_number'] = range(1, len(time_series) + 1)

# Monthly P&L
monthly_pnl = closed_positions.groupby(closed_positions['first_date'].dt.to_period('M')).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
monthly_pnl.columns = ['month', 'pnl', 'wins', 'trades', 'win_rate']

print("\nMONTHLY BREAKDOWN:")
print("-"*60)
for _, row in monthly_pnl.iterrows():
    status = "+" if row['pnl'] > 0 else ""
    print(f"  {row['month']}: {status}${row['pnl']:,.0f} | {row['win_rate']:.0%} win rate ({int(row['wins'])}/{int(row['trades'])} trades)")

# Best and worst months
best_month = monthly_pnl.loc[monthly_pnl['pnl'].idxmax()]
worst_month = monthly_pnl.loc[monthly_pnl['pnl'].idxmin()]
print(f"\nBest month: {best_month['month']} (+${best_month['pnl']:,.0f})")
print(f"Worst month: {worst_month['month']} (${worst_month['pnl']:,.0f})")


# =============================================================================
# 2. WINNING AND LOSING STREAKS
# =============================================================================
print("\n" + "="*70)
print("2. WINNING AND LOSING STREAKS")
print("="*70)

time_series = time_series.sort_values('last_date')
time_series['streak'] = (time_series['is_winner'] != time_series['is_winner'].shift()).cumsum()

streak_stats = time_series.groupby('streak').agg({
    'is_winner': 'first',
    'total_pnl': ['count', 'sum']
}).reset_index()
streak_stats.columns = ['streak_id', 'is_win_streak', 'length', 'pnl']

win_streaks = streak_stats[streak_stats['is_win_streak'] == True]
loss_streaks = streak_stats[streak_stats['is_win_streak'] == False]

print(f"\nLongest WIN streak: {int(win_streaks['length'].max())} trades (${win_streaks.loc[win_streaks['length'].idxmax(), 'pnl']:,.0f})")
print(f"Longest LOSS streak: {int(loss_streaks['length'].max())} trades (${loss_streaks.loc[loss_streaks['length'].idxmax(), 'pnl']:,.0f})")
print(f"Average win streak: {win_streaks['length'].mean():.1f} trades")
print(f"Average loss streak: {loss_streaks['length'].mean():.1f} trades")


# =============================================================================
# 3. POSITION SIZE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("3. POSITION SIZE ANALYSIS (by initial investment)")
print("="*70)

# Use absolute amount as proxy for position size
closed_positions['position_size'] = closed_positions['avg_price'] * abs(closed_positions['net_quantity'].replace(0, 1))

# Bucket by size
size_buckets = pd.qcut(closed_positions['position_size'].clip(lower=0.01), q=5, labels=['Tiny', 'Small', 'Medium', 'Large', 'XL'], duplicates='drop')
closed_positions['size_bucket'] = size_buckets

size_analysis = closed_positions.groupby('size_bucket').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'position_size': 'mean'
}).reset_index()
size_analysis.columns = ['size', 'pnl', 'wins', 'trades', 'win_rate', 'avg_size']

print("\nWin rate by POSITION SIZE:")
for _, row in size_analysis.iterrows():
    print(f"  {row['size']:8s} (avg ${row['avg_size']:,.0f}): {row['win_rate']:.0%} win rate | ${row['pnl']:,.0f}")


# =============================================================================
# 4. TICKER DEEP DIVE - TOP 10 MOST TRADED
# =============================================================================
print("\n" + "="*70)
print("4. TICKER DEEP DIVE - Your Most Traded Symbols")
print("="*70)

ticker_stats = closed_positions.groupby('underlying').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'hold_days': 'mean'
}).reset_index()
ticker_stats.columns = ['ticker', 'pnl', 'wins', 'trades', 'win_rate', 'avg_hold']
ticker_stats = ticker_stats.sort_values('trades', ascending=False)

print("\nTop 15 tickers by trade count:")
print("-"*70)
print(f"{'Ticker':<8} {'Trades':>7} {'Win%':>7} {'P&L':>12} {'Avg Hold':>10} {'Status':<10}")
print("-"*70)

for _, row in ticker_stats.head(15).iterrows():
    status = "WINNER" if row['pnl'] > 0 else "LOSER"
    color_pnl = f"+${row['pnl']:,.0f}" if row['pnl'] > 0 else f"${row['pnl']:,.0f}"
    print(f"{row['ticker']:<8} {int(row['trades']):>7} {row['win_rate']:>6.0%} {color_pnl:>12} {row['avg_hold']:>8.1f}d  {status:<10}")

# Biggest winners and losers by ticker
print("\n\nBIGGEST WINNING TICKERS (by total P&L):")
top_winners = ticker_stats.nlargest(5, 'pnl')
for _, row in top_winners.iterrows():
    print(f"  {row['ticker']}: +${row['pnl']:,.0f} ({row['win_rate']:.0%} win rate, {int(row['trades'])} trades)")

print("\nBIGGEST LOSING TICKERS (by total P&L):")
top_losers = ticker_stats.nsmallest(5, 'pnl')
for _, row in top_losers.iterrows():
    print(f"  {row['ticker']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win rate, {int(row['trades'])} trades)")


# =============================================================================
# 5. TIME OF MONTH ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("5. TIME OF MONTH ANALYSIS")
print("="*70)

closed_positions['day_of_month'] = closed_positions['first_date'].dt.day

# Early, mid, late month
def month_period(day):
    if day <= 10:
        return 'Early (1-10)'
    elif day <= 20:
        return 'Mid (11-20)'
    else:
        return 'Late (21-31)'

closed_positions['month_period'] = closed_positions['day_of_month'].apply(month_period)

month_period_stats = closed_positions.groupby('month_period').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
month_period_stats.columns = ['period', 'pnl', 'wins', 'trades', 'win_rate']

print("\nPerformance by TIME OF MONTH:")
for _, row in month_period_stats.iterrows():
    print(f"  {row['period']}: {row['win_rate']:.0%} win rate ({int(row['wins'])}/{int(row['trades'])}) | ${row['pnl']:,.0f}")


# =============================================================================
# 6. EXPIRATION ANALYSIS (for options)
# =============================================================================
print("\n" + "="*70)
print("6. EXPIRATION ANALYSIS")
print("="*70)

# Parse expiration from symbol (Fidelity format: -AAPL260116C280 = Jan 16, 2026, $280 call)
import re

def parse_option_details(symbol):
    if not isinstance(symbol, str) or not symbol.startswith('-'):
        return None, None, None

    # Pattern: -TICKER YYMMDD C/P STRIKE
    match = re.match(r'-([A-Z]+)(\d{6})([CP])(\d+\.?\d*)', symbol.replace(' ', ''))
    if match:
        ticker, date_str, opt_type, strike = match.groups()
        try:
            exp_date = datetime.strptime(date_str, '%y%m%d')
            return exp_date, opt_type, float(strike)
        except:
            pass
    return None, None, None

# Apply to closed positions
exp_data = closed_positions['Symbol'].apply(lambda x: pd.Series(parse_option_details(x)))
exp_data.columns = ['exp_date', 'opt_type_parsed', 'strike']
closed_positions = pd.concat([closed_positions, exp_data], axis=1)

# Calculate DTE (days to expiration) at entry
try:
    closed_positions['exp_date'] = pd.to_datetime(closed_positions['exp_date'])
    closed_positions['dte_at_entry'] = (closed_positions['exp_date'] - closed_positions['first_date']).dt.days
except:
    closed_positions['dte_at_entry'] = None

# Filter to valid DTE
dte_valid = closed_positions[closed_positions['dte_at_entry'].notna() & (closed_positions['dte_at_entry'] > 0)]

if len(dte_valid) > 20:
    # Bucket DTE
    dte_buckets = pd.cut(dte_valid['dte_at_entry'],
                         bins=[0, 7, 14, 30, 60, 365],
                         labels=['0-7 DTE', '8-14 DTE', '15-30 DTE', '31-60 DTE', '60+ DTE'])
    dte_valid = dte_valid.copy()
    dte_valid['dte_bucket'] = dte_buckets

    dte_stats = dte_valid.groupby('dte_bucket').agg({
        'total_pnl': 'sum',
        'is_winner': ['sum', 'count', 'mean']
    }).reset_index()
    dte_stats.columns = ['dte', 'pnl', 'wins', 'trades', 'win_rate']

    print("\nPerformance by DAYS TO EXPIRATION at entry:")
    for _, row in dte_stats.iterrows():
        print(f"  {row['dte']}: {row['win_rate']:.0%} win rate ({int(row['wins'])}/{int(row['trades'])}) | ${row['pnl']:,.0f}")
else:
    print("Not enough valid DTE data to analyze")


# =============================================================================
# 7. TRADE FREQUENCY ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("7. TRADE FREQUENCY ANALYSIS - Are You Overtrading?")
print("="*70)

# Trades per week
weekly_trades = closed_positions.groupby('week_of_year').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
weekly_trades.columns = ['week', 'pnl', 'wins', 'trades', 'win_rate']

# Correlation between trade count and performance
high_volume_weeks = weekly_trades[weekly_trades['trades'] > weekly_trades['trades'].median()]
low_volume_weeks = weekly_trades[weekly_trades['trades'] <= weekly_trades['trades'].median()]

print(f"\nAverage trades per week: {weekly_trades['trades'].mean():.1f}")
print(f"Max trades in a week: {int(weekly_trades['trades'].max())}")

print(f"\nHIGH VOLUME weeks (>{weekly_trades['trades'].median():.0f} trades):")
print(f"  Average win rate: {high_volume_weeks['win_rate'].mean():.0%}")
print(f"  Total P&L: ${high_volume_weeks['pnl'].sum():,.0f}")
print(f"  Avg P&L per week: ${high_volume_weeks['pnl'].mean():,.0f}")

print(f"\nLOW VOLUME weeks (<={weekly_trades['trades'].median():.0f} trades):")
print(f"  Average win rate: {low_volume_weeks['win_rate'].mean():.0%}")
print(f"  Total P&L: ${low_volume_weeks['pnl'].sum():,.0f}")
print(f"  Avg P&L per week: ${low_volume_weeks['pnl'].mean():,.0f}")


# =============================================================================
# 8. BEST AND WORST INDIVIDUAL TRADES
# =============================================================================
print("\n" + "="*70)
print("8. BEST AND WORST INDIVIDUAL TRADES")
print("="*70)

sorted_trades = closed_positions.sort_values('total_pnl', ascending=False)

print("\nTOP 10 BEST TRADES:")
print("-"*70)
for i, (_, row) in enumerate(sorted_trades.head(10).iterrows()):
    print(f"  {i+1}. +${row['total_pnl']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'} | {row['hold_days']}d hold | {row['account_type']}")

print("\nTOP 10 WORST TRADES:")
print("-"*70)
for i, (_, row) in enumerate(sorted_trades.tail(10).iloc[::-1].iterrows()):
    print(f"  {i+1}. ${row['total_pnl']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'} | {row['hold_days']}d hold | {row['account_type']}")


# =============================================================================
# 9. CALLS vs PUTS DEEP DIVE
# =============================================================================
print("\n" + "="*70)
print("9. CALLS vs PUTS - DETAILED BREAKDOWN")
print("="*70)

calls = closed_positions[closed_positions['option_type'] == 'CALL']
puts = closed_positions[closed_positions['option_type'] == 'PUT']

print("\nCALLS Analysis:")
print(f"  Total trades: {len(calls)}")
print(f"  Win rate: {calls['is_winner'].mean():.1%}")
print(f"  Total P&L: ${calls['total_pnl'].sum():,.0f}")
print(f"  Avg winner: ${calls[calls['is_winner']]['total_pnl'].mean():,.0f}")
print(f"  Avg loser: ${calls[~calls['is_winner']]['total_pnl'].mean():,.0f}")
print(f"  Avg hold time: {calls['hold_days'].mean():.1f} days")

print("\nPUTS Analysis:")
print(f"  Total trades: {len(puts)}")
print(f"  Win rate: {puts['is_winner'].mean():.1%}")
print(f"  Total P&L: ${puts['total_pnl'].sum():,.0f}")
print(f"  Avg winner: ${puts[puts['is_winner']]['total_pnl'].mean():,.0f}")
print(f"  Avg loser: ${puts[~puts['is_winner']]['total_pnl'].mean():,.0f}")
print(f"  Avg hold time: {puts['hold_days'].mean():.1f} days")

# Calls by underlying
print("\nCALLS - Best/Worst underlyings:")
call_by_ticker = calls.groupby('underlying').agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
call_by_ticker.columns = ['ticker', 'pnl', 'trades', 'win_rate']
call_by_ticker = call_by_ticker[call_by_ticker['trades'] >= 3]

print("  Best:")
for _, row in call_by_ticker.nlargest(3, 'pnl').iterrows():
    print(f"    {row['ticker']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win rate)")
print("  Worst:")
for _, row in call_by_ticker.nsmallest(3, 'pnl').iterrows():
    print(f"    {row['ticker']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win rate)")


# =============================================================================
# 10. DAY OF WEEK + OPTION TYPE COMBO
# =============================================================================
print("\n" + "="*70)
print("10. DAY OF WEEK + OPTION TYPE COMBINATIONS")
print("="*70)

day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

combo_stats = closed_positions.groupby(['day_of_week', 'option_type']).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
combo_stats.columns = ['day', 'opt_type', 'pnl', 'wins', 'trades', 'win_rate']
combo_stats = combo_stats[combo_stats['opt_type'].notna()]
combo_stats['day_name'] = combo_stats['day'].apply(lambda x: day_names[x])

print("\nBest day/option combinations:")
combo_stats_sorted = combo_stats.sort_values('pnl', ascending=False)
for _, row in combo_stats_sorted.head(5).iterrows():
    print(f"  {row['day_name']} {row['opt_type']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win, {int(row['trades'])} trades)")

print("\nWorst day/option combinations:")
for _, row in combo_stats_sorted.tail(5).iterrows():
    print(f"  {row['day_name']} {row['opt_type']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win, {int(row['trades'])} trades)")


# =============================================================================
# 11. CONSECUTIVE TRADE ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("11. DOES A WIN/LOSS AFFECT YOUR NEXT TRADE?")
print("="*70)

time_series = closed_positions.sort_values('first_date').copy()
time_series['prev_winner'] = time_series['is_winner'].shift(1)

# Performance after a win vs after a loss
after_win = time_series[time_series['prev_winner'] == True]
after_loss = time_series[time_series['prev_winner'] == False]

print(f"\nAfter a WIN:")
print(f"  Next trade win rate: {after_win['is_winner'].mean():.1%}")
print(f"  Avg P&L on next trade: ${after_win['total_pnl'].mean():,.0f}")

print(f"\nAfter a LOSS:")
print(f"  Next trade win rate: {after_loss['is_winner'].mean():.1%}")
print(f"  Avg P&L on next trade: ${after_loss['total_pnl'].mean():,.0f}")

if after_loss['is_winner'].mean() < after_win['is_winner'].mean():
    print("\n** WARNING: You perform worse after losses - possible tilt/revenge trading **")


# =============================================================================
# 12. ACCOUNT COMPARISON
# =============================================================================
print("\n" + "="*70)
print("12. ACCOUNT-BY-ACCOUNT BREAKDOWN")
print("="*70)

account_stats = closed_positions.groupby('account_type').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'hold_days': 'mean'
}).reset_index()
account_stats.columns = ['account', 'pnl', 'wins', 'trades', 'win_rate', 'avg_hold']

print("\nPerformance by ACCOUNT:")
print("-"*70)
for _, row in account_stats.sort_values('pnl', ascending=False).iterrows():
    status = "PROFIT" if row['pnl'] > 0 else "LOSS"
    print(f"  {row['account']:<12}: ${row['pnl']:>10,.0f} | {row['win_rate']:.0%} win | {int(row['trades'])} trades | {row['avg_hold']:.0f}d avg hold | {status}")


# =============================================================================
# 13. RISK/REWARD ANALYSIS
# =============================================================================
print("\n" + "="*70)
print("13. RISK/REWARD METRICS")
print("="*70)

winners = closed_positions[closed_positions['is_winner']]
losers = closed_positions[~closed_positions['is_winner']]

avg_win = winners['total_pnl'].mean()
avg_loss = abs(losers['total_pnl'].mean())
win_rate = len(winners) / len(closed_positions)

# Expected value per trade
expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

# Profit factor
total_wins = winners['total_pnl'].sum()
total_losses = abs(losers['total_pnl'].sum())
profit_factor = total_wins / total_losses if total_losses > 0 else 0

# Risk/reward ratio
risk_reward = avg_win / avg_loss if avg_loss > 0 else 0

print(f"\nKey Metrics:")
print(f"  Win Rate: {win_rate:.1%}")
print(f"  Average Win: ${avg_win:,.0f}")
print(f"  Average Loss: ${avg_loss:,.0f}")
print(f"  Risk/Reward Ratio: {risk_reward:.2f}:1")
print(f"  Profit Factor: {profit_factor:.2f}")
print(f"  Expected Value per Trade: ${expected_value:,.0f}")

print(f"\nTo break even at current R:R ({risk_reward:.2f}:1), you need {1/(1+risk_reward):.0%} win rate")
print(f"Your actual win rate: {win_rate:.0%}")

if expected_value < 0:
    # What win rate would you need?
    breakeven_wr = avg_loss / (avg_win + avg_loss)
    print(f"\n** To be profitable with current avg win/loss, you need {breakeven_wr:.0%} win rate **")


# =============================================================================
# 14. VOLATILITY ANALYSIS (by underlying)
# =============================================================================
print("\n" + "="*70)
print("14. WHICH TICKERS HAVE MOST VOLATILE RESULTS?")
print("="*70)

ticker_volatility = closed_positions.groupby('underlying').agg({
    'total_pnl': ['mean', 'std', 'count', 'sum']
}).reset_index()
ticker_volatility.columns = ['ticker', 'avg_pnl', 'std_pnl', 'trades', 'total_pnl']
ticker_volatility = ticker_volatility[ticker_volatility['trades'] >= 5]
ticker_volatility['consistency'] = ticker_volatility['avg_pnl'] / ticker_volatility['std_pnl'].replace(0, 1)

print("\nMost CONSISTENT winners (high avg, low volatility):")
consistent = ticker_volatility[ticker_volatility['avg_pnl'] > 0].nlargest(5, 'consistency')
for _, row in consistent.iterrows():
    print(f"  {row['ticker']}: ${row['avg_pnl']:,.0f} avg | ${row['std_pnl']:,.0f} std | {int(row['trades'])} trades")

print("\nMost INCONSISTENT (high volatility):")
inconsistent = ticker_volatility.nlargest(5, 'std_pnl')
for _, row in inconsistent.iterrows():
    print(f"  {row['ticker']}: ${row['avg_pnl']:,.0f} avg | ${row['std_pnl']:,.0f} std | {int(row['trades'])} trades")


# =============================================================================
# 15. FINAL SUMMARY & RECOMMENDATIONS
# =============================================================================
print("\n" + "="*70)
print("15. KEY INSIGHTS & RECOMMENDATIONS")
print("="*70)

insights = []

# Check calls vs puts
if calls['total_pnl'].sum() < -10000 and puts['total_pnl'].sum() > 0:
    insights.append(f"STOP BUYING CALLS - They lost you ${abs(calls['total_pnl'].sum()):,.0f}. Your puts made ${puts['total_pnl'].sum():,.0f}")

# Check hold duration
short_term = closed_positions[closed_positions['hold_days'] <= 1]
long_term = closed_positions[closed_positions['hold_days'] > 7]
if short_term['is_winner'].mean() < long_term['is_winner'].mean() and len(long_term) > 10:
    insights.append(f"HOLD LONGER - Short-term ({short_term['is_winner'].mean():.0%} win) vs Long-term ({long_term['is_winner'].mean():.0%} win)")

# Check overtrading
if high_volume_weeks['win_rate'].mean() < low_volume_weeks['win_rate'].mean():
    insights.append(f"TRADE LESS - High volume weeks: {high_volume_weeks['win_rate'].mean():.0%} win vs Low volume: {low_volume_weeks['win_rate'].mean():.0%}")

# Check after-loss performance
if after_loss['is_winner'].mean() < after_win['is_winner'].mean() - 0.05:
    insights.append(f"WATCH FOR TILT - You're {(after_win['is_winner'].mean() - after_loss['is_winner'].mean()):.0%} worse after losses")

# Best day
best_day_idx = closed_positions.groupby('day_of_week')['total_pnl'].sum().idxmax()
worst_day_idx = closed_positions.groupby('day_of_week')['total_pnl'].sum().idxmin()
insights.append(f"BEST DAY: {day_names[best_day_idx]} | WORST DAY: {day_names[worst_day_idx]}")

# Best ticker
best_ticker = ticker_stats.loc[ticker_stats['pnl'].idxmax(), 'ticker']
worst_ticker = ticker_stats.loc[ticker_stats['pnl'].idxmin(), 'ticker']
insights.append(f"BEST TICKER: {best_ticker} | WORST TICKER: {worst_ticker}")

print("\n" + "-"*70)
for i, insight in enumerate(insights, 1):
    print(f"{i}. {insight}")
print("-"*70)

print(f"\nOVERALL: {len(closed_positions)} trades | {win_rate:.0%} win rate | ${closed_positions['total_pnl'].sum():,.0f} total P&L")
print(f"Expected value per trade: ${expected_value:,.0f}")

if expected_value < 0:
    print(f"\n** Your system has NEGATIVE expected value. Every trade costs you ~${abs(expected_value):.0f} on average. **")
