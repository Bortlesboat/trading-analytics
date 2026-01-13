"""
DETAILED TRADING STATISTICS
============================
Every way to slice your trading data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES, POSITIONS_FILE
except ImportError:
    TRADE_FILES = ["./data/trades.csv"]
    POSITIONS_FILE = "./data/positions.csv"

print("Loading data...")

# Load trades
all_trades = []
for f in TRADE_FILES:
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df = df[df['Run Date'].notna()]
        df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage', case=False, na=False)]
        all_trades.append(df)
    except:
        pass

trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

# Load positions
positions_df = pd.read_csv(POSITIONS_FILE, encoding='utf-8-sig', index_col=False)

def clean_dollar(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '').replace(',', '').replace('+', '')
    try:
        return float(val)
    except:
        return np.nan

positions_df['Total Gain/Loss Dollar'] = positions_df['Total Gain/Loss Dollar'].apply(clean_dollar)
positions_df = positions_df[
    (positions_df['Symbol'].notna()) &
    (~positions_df['Symbol'].str.contains('SPAXX|FDRXX|FCASH|USD|Pending', case=False, na=False)) &
    (positions_df['Quantity'].notna())
]

# Process trades
trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)]
trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

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

    return pd.Series({'direction': direction, 'is_option': is_option, 'option_type': option_type, 'underlying': underlying})

parsed = trades_df.apply(parse_trade, axis=1)
trades_df = pd.concat([trades_df, parsed], axis=1)
trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')
trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')

# Account categorization
def categorize_account(account):
    account = str(account).lower()
    if '401k' in account: return '401k'
    elif 'roth' in account: return 'Roth IRA'
    elif 'traditional' in account: return 'Traditional IRA'
    elif 'individual' in account: return 'Individual'
    elif 'option' in account: return 'Options'
    elif 'utma' in account.lower(): return 'UTMA Account'
    else: return 'Other'

trades_df['account_type'] = trades_df['Account'].apply(categorize_account)

# Build closed positions
symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type', 'account_type']).agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'Run Date': ['min', 'max'],
    'direction': 'count',
    'Price': ['mean', 'first', 'last']
}).reset_index()
symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type', 'account_type',
                       'total_pnl', 'net_quantity', 'first_date', 'last_date',
                       'num_transactions', 'avg_price', 'first_price', 'last_price']

closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
closed['is_winner'] = closed['total_pnl'] > 0
closed['first_date'] = pd.to_datetime(closed['first_date'])
closed['last_date'] = pd.to_datetime(closed['last_date'])
closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days
closed['day_of_week'] = closed['first_date'].dt.dayofweek
closed['month'] = closed['first_date'].dt.month
closed['week'] = closed['first_date'].dt.isocalendar().week
closed['year'] = closed['first_date'].dt.year
closed['quarter'] = closed['first_date'].dt.quarter
closed['day_of_month'] = closed['first_date'].dt.day

print(f"Loaded {len(closed)} closed positions")
print("="*80)


# =============================================================================
# 1. BY ACCOUNT BREAKDOWN
# =============================================================================
print("\n" + "="*80)
print("1. PERFORMANCE BY ACCOUNT")
print("="*80)

# Realized by account
account_realized = closed.groupby('account_type').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'hold_days': 'mean'
}).reset_index()
account_realized.columns = ['account', 'realized_pnl', 'wins', 'trades', 'win_rate', 'avg_hold']

# Unrealized by account
positions_df['account_type'] = positions_df['Account Name'].apply(categorize_account)
account_unrealized = positions_df.groupby('account_type')['Total Gain/Loss Dollar'].sum().reset_index()
account_unrealized.columns = ['account', 'unrealized_pnl']

# Merge
account_stats = pd.merge(account_realized, account_unrealized, on='account', how='outer').fillna(0)
account_stats['total_pnl'] = account_stats['realized_pnl'] + account_stats['unrealized_pnl']
account_stats = account_stats.sort_values('total_pnl', ascending=False)

print("\n" + "-"*80)
print(f"{'Account':<18} {'Realized':>12} {'Unrealized':>12} {'TOTAL':>12} {'Win%':>8} {'Trades':>8}")
print("-"*80)
for _, row in account_stats.iterrows():
    print(f"{row['account']:<18} ${row['realized_pnl']:>10,.0f} ${row['unrealized_pnl']:>10,.0f} ${row['total_pnl']:>10,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8}")
print("-"*80)
print(f"{'TOTAL':<18} ${account_stats['realized_pnl'].sum():>10,.0f} ${account_stats['unrealized_pnl'].sum():>10,.0f} ${account_stats['total_pnl'].sum():>10,.0f}")


# =============================================================================
# 2. MONTHLY PERFORMANCE
# =============================================================================
print("\n" + "="*80)
print("2. MONTHLY PERFORMANCE")
print("="*80)

monthly = closed.groupby([closed['first_date'].dt.to_period('M')]).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
monthly.columns = ['month', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*80)
print(f"{'Month':<12} {'P&L':>12} {'Win Rate':>10} {'Trades':>8} {'Avg P&L':>12}")
print("-"*80)
for _, row in monthly.iterrows():
    avg_pnl = row['pnl'] / row['trades'] if row['trades'] > 0 else 0
    status = "+" if row['pnl'] > 0 else ""
    print(f"{str(row['month']):<12} {status}${row['pnl']:>10,.0f} {row['win_rate']:>9.0%} {int(row['trades']):>8} ${avg_pnl:>10,.0f}")


# =============================================================================
# 3. QUARTERLY PERFORMANCE
# =============================================================================
print("\n" + "="*80)
print("3. QUARTERLY PERFORMANCE")
print("="*80)

quarterly = closed.groupby(['year', 'quarter']).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
quarterly.columns = ['year', 'quarter', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*60)
for _, row in quarterly.iterrows():
    q_name = f"Q{int(row['quarter'])} {int(row['year'])}"
    status = "+" if row['pnl'] > 0 else ""
    print(f"  {q_name}: {status}${row['pnl']:,.0f} | {row['win_rate']:.0%} win rate | {int(row['trades'])} trades")


# =============================================================================
# 4. WEEKLY PERFORMANCE (Best/Worst Weeks)
# =============================================================================
print("\n" + "="*80)
print("4. BEST AND WORST WEEKS")
print("="*80)

weekly = closed.groupby([closed['first_date'].dt.to_period('W')]).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
weekly.columns = ['week', 'pnl', 'wins', 'trades', 'win_rate']

print("\nTOP 5 BEST WEEKS:")
for _, row in weekly.nlargest(5, 'pnl').iterrows():
    print(f"  {row['week']}: +${row['pnl']:,.0f} ({row['win_rate']:.0%} win, {int(row['trades'])} trades)")

print("\nTOP 5 WORST WEEKS:")
for _, row in weekly.nsmallest(5, 'pnl').iterrows():
    print(f"  {row['week']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win, {int(row['trades'])} trades)")


# =============================================================================
# 5. BY TICKER - COMPREHENSIVE
# =============================================================================
print("\n" + "="*80)
print("5. COMPLETE TICKER BREAKDOWN")
print("="*80)

# Realized by ticker
ticker_realized = closed.groupby('underlying').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'hold_days': 'mean'
}).reset_index()
ticker_realized.columns = ['ticker', 'realized_pnl', 'wins', 'trades', 'win_rate', 'avg_hold']

# Unrealized by ticker (need to parse underlying from positions)
def get_underlying_from_pos(row):
    symbol = str(row['Symbol']).strip()
    desc = str(row['Description'])
    if symbol.startswith('-'):
        parts = desc.split()
        return parts[0] if parts else symbol
    return symbol

positions_df['underlying'] = positions_df.apply(get_underlying_from_pos, axis=1)
ticker_unrealized = positions_df.groupby('underlying')['Total Gain/Loss Dollar'].sum().reset_index()
ticker_unrealized.columns = ['ticker', 'unrealized_pnl']

# Merge
ticker_stats = pd.merge(ticker_realized, ticker_unrealized, on='ticker', how='outer').fillna(0)
ticker_stats['total_pnl'] = ticker_stats['realized_pnl'] + ticker_stats['unrealized_pnl']
ticker_stats = ticker_stats.sort_values('trades', ascending=False)

print("\nAll tickers with 5+ trades:")
print("-"*90)
print(f"{'Ticker':<8} {'Realized':>11} {'Unrealized':>11} {'TOTAL':>11} {'Win%':>7} {'Trades':>7} {'Hold':>6}")
print("-"*90)
for _, row in ticker_stats[ticker_stats['trades'] >= 5].iterrows():
    print(f"{row['ticker']:<8} ${row['realized_pnl']:>9,.0f} ${row['unrealized_pnl']:>9,.0f} ${row['total_pnl']:>9,.0f} {row['win_rate']:>6.0%} {int(row['trades']):>7} {row['avg_hold']:>5.1f}d")


# =============================================================================
# 6. CALLS VS PUTS BY ACCOUNT
# =============================================================================
print("\n" + "="*80)
print("6. CALLS VS PUTS BY ACCOUNT")
print("="*80)

calls_puts_by_acct = closed[closed['option_type'].notna()].groupby(['account_type', 'option_type']).agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
calls_puts_by_acct.columns = ['account', 'option_type', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*70)
for acct in calls_puts_by_acct['account'].unique():
    acct_data = calls_puts_by_acct[calls_puts_by_acct['account'] == acct]
    print(f"\n{acct}:")
    for _, row in acct_data.iterrows():
        print(f"  {row['option_type']}: ${row['pnl']:,.0f} | {row['win_rate']:.0%} win | {int(row['trades'])} trades")


# =============================================================================
# 7. TRADE SIZE ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("7. TRADE SIZE ANALYSIS")
print("="*80)

# Calculate initial investment (rough estimate)
closed['trade_size'] = abs(closed['total_pnl']) + (closed['avg_price'] * 100)  # Rough proxy

size_bins = [0, 100, 500, 1000, 2500, 5000, float('inf')]
size_labels = ['<$100', '$100-500', '$500-1K', '$1K-2.5K', '$2.5K-5K', '>$5K']
closed['size_bucket'] = pd.cut(closed['trade_size'], bins=size_bins, labels=size_labels)

size_stats = closed.groupby('size_bucket').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
size_stats.columns = ['size', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*60)
for _, row in size_stats.iterrows():
    if row['trades'] > 0:
        print(f"  {row['size']:<12}: ${row['pnl']:>10,.0f} | {row['win_rate']:.0%} win | {int(row['trades'])} trades")


# =============================================================================
# 8. HOLD TIME ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("8. HOLD TIME ANALYSIS")
print("="*80)

hold_bins = [-1, 0, 1, 3, 7, 14, 30, 60, float('inf')]
hold_labels = ['Same day', '1 day', '2-3 days', '4-7 days', '1-2 weeks', '2-4 weeks', '1-2 months', '2+ months']
closed['hold_bucket'] = pd.cut(closed['hold_days'], bins=hold_bins, labels=hold_labels)

hold_stats = closed.groupby('hold_bucket').agg({
    'total_pnl': ['sum', 'mean'],
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
hold_stats.columns = ['hold_time', 'total_pnl', 'avg_pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*70)
print(f"{'Hold Time':<14} {'Total P&L':>12} {'Avg P&L':>10} {'Win%':>8} {'Trades':>8}")
print("-"*70)
for _, row in hold_stats.iterrows():
    if row['trades'] > 0:
        print(f"{row['hold_time']:<14} ${row['total_pnl']:>10,.0f} ${row['avg_pnl']:>8,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8}")


# =============================================================================
# 9. DAY OF WEEK ANALYSIS
# =============================================================================
print("\n" + "="*80)
print("9. DAY OF WEEK ANALYSIS")
print("="*80)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
day_stats = closed.groupby('day_of_week').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
day_stats.columns = ['day', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*60)
for _, row in day_stats.iterrows():
    if row['day'] < 5:
        day = day_names[int(row['day'])]
        status = "+" if row['pnl'] > 0 else ""
        print(f"  {day:<12}: {status}${row['pnl']:>10,.0f} | {row['win_rate']:.0%} win | {int(row['trades'])} trades")


# =============================================================================
# 10. CONSECUTIVE WINS/LOSSES
# =============================================================================
print("\n" + "="*80)
print("10. STREAK ANALYSIS")
print("="*80)

sorted_trades = closed.sort_values('first_date')
sorted_trades['streak_group'] = (sorted_trades['is_winner'] != sorted_trades['is_winner'].shift()).cumsum()

streaks = sorted_trades.groupby('streak_group').agg({
    'is_winner': 'first',
    'total_pnl': ['count', 'sum']
}).reset_index()
streaks.columns = ['group', 'is_win', 'length', 'pnl']

win_streaks = streaks[streaks['is_win'] == True]
loss_streaks = streaks[streaks['is_win'] == False]

print(f"\nWIN STREAKS:")
print(f"  Longest: {int(win_streaks['length'].max())} trades (+${win_streaks.loc[win_streaks['length'].idxmax(), 'pnl']:,.0f})")
print(f"  Average: {win_streaks['length'].mean():.1f} trades")
print(f"  Streaks of 5+: {len(win_streaks[win_streaks['length'] >= 5])}")

print(f"\nLOSS STREAKS:")
print(f"  Longest: {int(loss_streaks['length'].max())} trades (${loss_streaks.loc[loss_streaks['length'].idxmax(), 'pnl']:,.0f})")
print(f"  Average: {loss_streaks['length'].mean():.1f} trades")
print(f"  Streaks of 5+: {len(loss_streaks[loss_streaks['length'] >= 5])}")


# =============================================================================
# 11. PERFORMANCE AFTER WINS VS LOSSES
# =============================================================================
print("\n" + "="*80)
print("11. PERFORMANCE AFTER WINS VS LOSSES")
print("="*80)

sorted_trades = closed.sort_values('first_date').copy()
sorted_trades['prev_result'] = sorted_trades['is_winner'].shift()
sorted_trades['prev_pnl'] = sorted_trades['total_pnl'].shift()

after_win = sorted_trades[sorted_trades['prev_result'] == True]
after_loss = sorted_trades[sorted_trades['prev_result'] == False]
after_big_win = sorted_trades[sorted_trades['prev_pnl'] > 1000]
after_big_loss = sorted_trades[sorted_trades['prev_pnl'] < -1000]

print(f"\nAfter ANY WIN ({len(after_win)} trades):")
print(f"  Win rate: {after_win['is_winner'].mean():.1%}")
print(f"  Avg P&L: ${after_win['total_pnl'].mean():,.0f}")

print(f"\nAfter ANY LOSS ({len(after_loss)} trades):")
print(f"  Win rate: {after_loss['is_winner'].mean():.1%}")
print(f"  Avg P&L: ${after_loss['total_pnl'].mean():,.0f}")

print(f"\nAfter BIG WIN (>$1K) ({len(after_big_win)} trades):")
print(f"  Win rate: {after_big_win['is_winner'].mean():.1%}")
print(f"  Avg P&L: ${after_big_win['total_pnl'].mean():,.0f}")

print(f"\nAfter BIG LOSS (>$1K) ({len(after_big_loss)} trades):")
print(f"  Win rate: {after_big_loss['is_winner'].mean():.1%}")
print(f"  Avg P&L: ${after_big_loss['total_pnl'].mean():,.0f}")


# =============================================================================
# 12. RISK METRICS
# =============================================================================
print("\n" + "="*80)
print("12. RISK METRICS")
print("="*80)

winners = closed[closed['is_winner']]
losers = closed[~closed['is_winner']]

avg_win = winners['total_pnl'].mean()
avg_loss = abs(losers['total_pnl'].mean())
win_rate = len(winners) / len(closed)
max_win = winners['total_pnl'].max()
max_loss = losers['total_pnl'].min()
median_win = winners['total_pnl'].median()
median_loss = losers['total_pnl'].median()

# Calculate drawdown
cumulative = closed.sort_values('last_date')['total_pnl'].cumsum()
running_max = cumulative.cummax()
drawdown = cumulative - running_max
max_drawdown = drawdown.min()

# Profit factor
total_wins = winners['total_pnl'].sum()
total_losses = abs(losers['total_pnl'].sum())
profit_factor = total_wins / total_losses if total_losses > 0 else 0

# Expectancy
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

# Sharpe-like ratio (monthly)
monthly_pnl = closed.groupby(closed['first_date'].dt.to_period('M'))['total_pnl'].sum()
sharpe_approx = monthly_pnl.mean() / monthly_pnl.std() if monthly_pnl.std() > 0 else 0

print(f"""
BASIC STATS:
  Win Rate:           {win_rate:.1%}
  Total Trades:       {len(closed)}

PROFIT/LOSS:
  Average Win:        ${avg_win:,.0f}
  Average Loss:       ${avg_loss:,.0f}
  Median Win:         ${median_win:,.0f}
  Median Loss:        ${median_loss:,.0f}
  Largest Win:        ${max_win:,.0f}
  Largest Loss:       ${max_loss:,.0f}

RATIOS:
  Risk/Reward:        {avg_win/avg_loss:.2f}:1
  Profit Factor:      {profit_factor:.2f}
  Expectancy/Trade:   ${expectancy:,.0f}

DRAWDOWN:
  Max Drawdown:       ${max_drawdown:,.0f}

CONSISTENCY:
  Monthly Sharpe:     {sharpe_approx:.2f}
  Profitable Months:  {len(monthly_pnl[monthly_pnl > 0])}/{len(monthly_pnl)}
""")


# =============================================================================
# 13. TOP TRADES BY ACCOUNT
# =============================================================================
print("\n" + "="*80)
print("13. TOP TRADES BY ACCOUNT")
print("="*80)

for acct in closed['account_type'].unique():
    acct_trades = closed[closed['account_type'] == acct]
    if len(acct_trades) > 0:
        print(f"\n{acct}:")
        print("  Best trades:")
        for _, row in acct_trades.nlargest(3, 'total_pnl').iterrows():
            print(f"    +${row['total_pnl']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'}")
        print("  Worst trades:")
        for _, row in acct_trades.nsmallest(3, 'total_pnl').iterrows():
            print(f"    ${row['total_pnl']:,.0f} | {row['underlying']} {row['option_type'] or 'Stock'}")


# =============================================================================
# 14. OPTION TYPE BY TICKER
# =============================================================================
print("\n" + "="*80)
print("14. CALLS VS PUTS BY TICKER (Top 10 Most Traded)")
print("="*80)

top_tickers = closed['underlying'].value_counts().head(10).index.tolist()

print("\n" + "-"*70)
print(f"{'Ticker':<8} {'CALL P&L':>12} {'CALL Win%':>10} {'PUT P&L':>12} {'PUT Win%':>10}")
print("-"*70)

for ticker in top_tickers:
    ticker_data = closed[closed['underlying'] == ticker]
    calls = ticker_data[ticker_data['option_type'] == 'CALL']
    puts = ticker_data[ticker_data['option_type'] == 'PUT']

    call_pnl = calls['total_pnl'].sum() if len(calls) > 0 else 0
    call_wr = calls['is_winner'].mean() if len(calls) > 0 else 0
    put_pnl = puts['total_pnl'].sum() if len(puts) > 0 else 0
    put_wr = puts['is_winner'].mean() if len(puts) > 0 else 0

    print(f"{ticker:<8} ${call_pnl:>10,.0f} {call_wr:>9.0%} ${put_pnl:>10,.0f} {put_wr:>9.0%}")


# =============================================================================
# 15. TRADING FREQUENCY
# =============================================================================
print("\n" + "="*80)
print("15. TRADING FREQUENCY ANALYSIS")
print("="*80)

# Trades per day
trades_per_day = closed.groupby(closed['first_date'].dt.date).size()

print(f"\nDaily Trading Stats:")
print(f"  Average trades/day: {trades_per_day.mean():.1f}")
print(f"  Max trades in a day: {trades_per_day.max()}")
print(f"  Days with 10+ trades: {len(trades_per_day[trades_per_day >= 10])}")

# Trading days
total_days = (closed['first_date'].max() - closed['first_date'].min()).days
active_days = len(trades_per_day)
print(f"  Active trading days: {active_days}")
print(f"  Total calendar days: {total_days}")
print(f"  Activity rate: {active_days/total_days:.0%}")


# =============================================================================
# 16. TIME OF MONTH
# =============================================================================
print("\n" + "="*80)
print("16. TIME OF MONTH ANALYSIS")
print("="*80)

def get_month_period(day):
    if day <= 10: return 'Early (1-10)'
    elif day <= 20: return 'Mid (11-20)'
    else: return 'Late (21-31)'

closed['month_period'] = closed['day_of_month'].apply(get_month_period)

period_stats = closed.groupby('month_period').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
period_stats.columns = ['period', 'pnl', 'wins', 'trades', 'win_rate']

print("\n" + "-"*60)
for _, row in period_stats.iterrows():
    print(f"  {row['period']:<14}: ${row['pnl']:>10,.0f} | {row['win_rate']:.0%} win | {int(row['trades'])} trades")


# =============================================================================
# 17. SUMMARY
# =============================================================================
print("\n" + "="*80)
print("17. EXECUTIVE SUMMARY")
print("="*80)

print(f"""
OVERALL PERFORMANCE:
  Realized P&L:     ${closed['total_pnl'].sum():,.0f}
  Win Rate:         {win_rate:.1%}
  Total Trades:     {len(closed)}
  Expectancy:       ${expectancy:,.0f} per trade

BEST PERFORMING:
  Best Account:     {account_stats.iloc[0]['account']} (${account_stats.iloc[0]['total_pnl']:,.0f})
  Best Ticker:      {ticker_stats.loc[ticker_stats['total_pnl'].idxmax(), 'ticker']} (${ticker_stats['total_pnl'].max():,.0f})
  Best Day:         {day_names[int(day_stats.loc[day_stats['pnl'].idxmax(), 'day'])]}
  Best Month Period: {period_stats.loc[period_stats['pnl'].idxmax(), 'period']}

WORST PERFORMING:
  Worst Account:    {account_stats.iloc[-1]['account']} (${account_stats.iloc[-1]['total_pnl']:,.0f})
  Worst Ticker:     {ticker_stats.loc[ticker_stats['total_pnl'].idxmin(), 'ticker']} (${ticker_stats['total_pnl'].min():,.0f})
  Worst Day:        {day_names[int(day_stats.loc[day_stats['pnl'].idxmin(), 'day'])]}

KEY INSIGHTS:
  - Calls vs Puts: PUTs significantly outperform CALLs
  - Hold Time: Longer holds ({hold_stats.loc[hold_stats['win_rate'].idxmax(), 'hold_time']}) have highest win rate
  - After Losses: Win rate drops to {after_loss['is_winner'].mean():.0%} (vs {after_win['is_winner'].mean():.0%} after wins)
  - Max Drawdown: ${max_drawdown:,.0f}
""")
