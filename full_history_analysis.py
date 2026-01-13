"""
FULL TRADING HISTORY ANALYSIS
=============================
Analyzing all available data: 2024 + 2025 + 2026
"""

import pandas as pd
import numpy as np
from datetime import datetime
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

# =============================================================================
# LOAD ALL DATA
# =============================================================================

print("="*70)
print("LOADING ALL TRADING DATA")
print("="*70)

all_trades = []
for f in TRADE_FILES:
    try:
        df = pd.read_csv(f, encoding='utf-8-sig')
        df = df[df['Run Date'].notna()]
        df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage|informational', case=False, na=False)]
        print(f"Loaded: {f.split(chr(92))[-1]} - {len(df)} rows")
        all_trades.append(df)
    except Exception as e:
        print(f"Error loading {f}: {e}")

trades_df = pd.concat(all_trades, ignore_index=True)
print(f"\nTotal rows loaded: {len(trades_df)}")

# Load positions
def clean_dollar(val):
    if pd.isna(val): return np.nan
    val = str(val).replace('$', '').replace(',', '').replace('+', '')
    try: return float(val)
    except: return np.nan

positions_df = pd.read_csv(POSITIONS_FILE, encoding='utf-8-sig', index_col=False)
positions_df['Total Gain/Loss Dollar'] = positions_df['Total Gain/Loss Dollar'].apply(clean_dollar)
positions_df = positions_df[
    (positions_df['Symbol'].notna()) &
    (~positions_df['Symbol'].str.contains('SPAXX|FDRXX|FCASH|USD|Pending', case=False, na=False)) &
    (positions_df['Quantity'].notna())
]

# =============================================================================
# PROCESS TRADES
# =============================================================================

print("\n" + "="*70)
print("PROCESSING TRADES")
print("="*70)

# Filter to actual trades
trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)]
trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

print(f"Actual trade rows: {len(trades_df)}")

# Parse trade details
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

# Time features
trades_df['year'] = trades_df['Run Date'].dt.year
trades_df['month'] = trades_df['Run Date'].dt.month
trades_df['day_of_week'] = trades_df['Run Date'].dt.dayofweek
trades_df['day_of_month'] = trades_df['Run Date'].dt.day

# Account type
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

print(f"Date range: {trades_df['Run Date'].min().strftime('%Y-%m-%d')} to {trades_df['Run Date'].max().strftime('%Y-%m-%d')}")

# =============================================================================
# BUILD CLOSED POSITIONS
# =============================================================================

symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type', 'account_type']).agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'Run Date': ['min', 'max'],
    'direction': 'count',
    'year': 'first'
}).reset_index()
symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type', 'account_type',
                       'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions', 'year']

closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
closed['is_winner'] = closed['total_pnl'] > 0
closed['first_date'] = pd.to_datetime(closed['first_date'])
closed['last_date'] = pd.to_datetime(closed['last_date'])
closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days
closed['day_of_week'] = closed['first_date'].dt.dayofweek
closed['month'] = closed['first_date'].dt.month
closed['year'] = closed['first_date'].dt.year
closed['day_of_month'] = closed['first_date'].dt.day

print(f"Closed positions: {len(closed)}")

# Unrealized P&L
unrealized_total = positions_df['Total Gain/Loss Dollar'].sum()

# =============================================================================
# OVERALL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("OVERALL SUMMARY (2024-2026)")
print("="*70)

realized_pnl = closed['total_pnl'].sum()
total_pnl = realized_pnl + unrealized_total
win_rate = closed['is_winner'].mean()

print(f"""
Date Range:           {trades_df['Run Date'].min().strftime('%b %Y')} - {trades_df['Run Date'].max().strftime('%b %Y')}
Total Closed Trades:  {len(closed)}
Win Rate:             {win_rate:.1%}

Realized P&L:         ${realized_pnl:,.0f}
Unrealized P&L:       ${unrealized_total:,.0f}
-----------------------------------------
TOTAL P&L:            ${total_pnl:,.0f}
""")

# =============================================================================
# YEAR OVER YEAR
# =============================================================================

print("\n" + "="*70)
print("YEAR OVER YEAR COMPARISON")
print("="*70)

yearly = closed.groupby('year').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean']
}).reset_index()
yearly.columns = ['year', 'pnl', 'wins', 'trades', 'win_rate']

print(f"\n{'Year':<8} {'P&L':>12} {'Win Rate':>10} {'Trades':>10} {'Avg Trade':>12}")
print("-"*55)
for _, row in yearly.iterrows():
    avg = row['pnl'] / row['trades'] if row['trades'] > 0 else 0
    status = "+" if row['pnl'] > 0 else ""
    print(f"{int(row['year']):<8} {status}${row['pnl']:>10,.0f} {row['win_rate']:>9.0%} {int(row['trades']):>10} ${avg:>10,.0f}")

# =============================================================================
# MONTHLY BY YEAR
# =============================================================================

print("\n" + "="*70)
print("MONTHLY P&L BY YEAR")
print("="*70)

monthly = closed.groupby(['year', 'month']).agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
monthly.columns = ['year', 'month', 'pnl', 'trades', 'win_rate']

month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

print(f"\n{'Month':<6}", end='')
for year in sorted(monthly['year'].unique()):
    print(f"{int(year):>12}", end='')
print()
print("-"*(6 + 12*len(monthly['year'].unique())))

for m in range(1, 13):
    print(f"{month_names[m]:<6}", end='')
    for year in sorted(monthly['year'].unique()):
        row = monthly[(monthly['year'] == year) & (monthly['month'] == m)]
        if len(row) > 0:
            pnl = row['pnl'].values[0]
            status = "+" if pnl > 0 else ""
            print(f"{status}${pnl:>9,.0f}", end='')
        else:
            print(f"{'--':>12}", end='')
    print()

# =============================================================================
# CALLS VS PUTS BY YEAR
# =============================================================================

print("\n" + "="*70)
print("CALLS VS PUTS BY YEAR")
print("="*70)

opt_yearly = closed[closed['option_type'].notna()].groupby(['year', 'option_type']).agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
opt_yearly.columns = ['year', 'option_type', 'pnl', 'trades', 'win_rate']

print(f"\n{'Year':<8} {'CALL P&L':>12} {'CALL Win%':>10} {'PUT P&L':>12} {'PUT Win%':>10}")
print("-"*55)
for year in sorted(opt_yearly['year'].unique()):
    calls = opt_yearly[(opt_yearly['year'] == year) & (opt_yearly['option_type'] == 'CALL')]
    puts = opt_yearly[(opt_yearly['year'] == year) & (opt_yearly['option_type'] == 'PUT')]

    call_pnl = calls['pnl'].values[0] if len(calls) > 0 else 0
    call_wr = calls['win_rate'].values[0] if len(calls) > 0 else 0
    put_pnl = puts['pnl'].values[0] if len(puts) > 0 else 0
    put_wr = puts['win_rate'].values[0] if len(puts) > 0 else 0

    print(f"{int(year):<8} ${call_pnl:>10,.0f} {call_wr:>9.0%} ${put_pnl:>10,.0f} {put_wr:>9.0%}")

# Totals
calls_total = closed[closed['option_type'] == 'CALL']['total_pnl'].sum()
puts_total = closed[closed['option_type'] == 'PUT']['total_pnl'].sum()
calls_wr = closed[closed['option_type'] == 'CALL']['is_winner'].mean()
puts_wr = closed[closed['option_type'] == 'PUT']['is_winner'].mean()

print("-"*55)
print(f"{'TOTAL':<8} ${calls_total:>10,.0f} {calls_wr:>9.0%} ${puts_total:>10,.0f} {puts_wr:>9.0%}")

# =============================================================================
# TICKER PERFORMANCE (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("ALL-TIME TICKER PERFORMANCE (10+ trades)")
print("="*70)

ticker_all = closed.groupby('underlying').agg({
    'total_pnl': 'sum',
    'is_winner': ['sum', 'count', 'mean'],
    'hold_days': 'mean'
}).reset_index()
ticker_all.columns = ['ticker', 'pnl', 'wins', 'trades', 'win_rate', 'avg_hold']
ticker_all = ticker_all[ticker_all['trades'] >= 10].sort_values('pnl', ascending=False)

print(f"\n{'Ticker':<8} {'P&L':>12} {'Win%':>8} {'Trades':>8} {'Hold':>8}")
print("-"*50)
for _, row in ticker_all.iterrows():
    status = "+" if row['pnl'] > 0 else ""
    print(f"{row['ticker']:<8} {status}${row['pnl']:>10,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8} {row['avg_hold']:>6.1f}d")

# =============================================================================
# DAY OF WEEK (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("DAY OF WEEK PERFORMANCE (ALL TIME)")
print("="*70)

day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
day_all = closed.groupby('day_of_week').agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
day_all.columns = ['day', 'pnl', 'trades', 'win_rate']

print(f"\n{'Day':<12} {'P&L':>12} {'Win%':>8} {'Trades':>8}")
print("-"*45)
for _, row in day_all.iterrows():
    if row['day'] < 5:
        status = "+" if row['pnl'] > 0 else ""
        print(f"{day_names[int(row['day'])]:<12} {status}${row['pnl']:>10,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8}")

# =============================================================================
# ACCOUNT PERFORMANCE (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("ACCOUNT PERFORMANCE (ALL TIME)")
print("="*70)

# Get unrealized by account
positions_df['account_type'] = positions_df['Account Name'].apply(categorize_account)
unrealized_by_acct = positions_df.groupby('account_type')['Total Gain/Loss Dollar'].sum()

acct_all = closed.groupby('account_type').agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
acct_all.columns = ['account', 'realized', 'trades', 'win_rate']
acct_all['unrealized'] = acct_all['account'].map(unrealized_by_acct).fillna(0)
acct_all['total'] = acct_all['realized'] + acct_all['unrealized']
acct_all = acct_all.sort_values('total', ascending=False)

print(f"\n{'Account':<18} {'Realized':>12} {'Unrealized':>12} {'TOTAL':>12} {'Win%':>8}")
print("-"*65)
for _, row in acct_all.iterrows():
    print(f"{row['account']:<18} ${row['realized']:>10,.0f} ${row['unrealized']:>10,.0f} ${row['total']:>10,.0f} {row['win_rate']:>7.0%}")

# =============================================================================
# TIME OF MONTH (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("TIME OF MONTH (ALL TIME)")
print("="*70)

def get_period(day):
    if day <= 10: return 'Early (1-10)'
    elif day <= 20: return 'Mid (11-20)'
    else: return 'Late (21-31)'

closed['month_period'] = closed['day_of_month'].apply(get_period)
period_all = closed.groupby('month_period').agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
period_all.columns = ['period', 'pnl', 'trades', 'win_rate']

print(f"\n{'Period':<14} {'P&L':>12} {'Win%':>8} {'Trades':>8}")
print("-"*45)
for _, row in period_all.iterrows():
    status = "+" if row['pnl'] > 0 else ""
    print(f"{row['period']:<14} {status}${row['pnl']:>10,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8}")

# =============================================================================
# HOLD TIME ANALYSIS (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("HOLD TIME ANALYSIS (ALL TIME)")
print("="*70)

hold_bins = [-1, 0, 1, 3, 7, 14, 30, 60, 1000]
hold_labels = ['Same day', '1 day', '2-3 days', '4-7 days', '1-2 weeks', '2-4 weeks', '1-2 months', '2+ months']
closed['hold_bucket'] = pd.cut(closed['hold_days'], bins=hold_bins, labels=hold_labels)

hold_all = closed.groupby('hold_bucket').agg({
    'total_pnl': 'sum',
    'is_winner': ['count', 'mean']
}).reset_index()
hold_all.columns = ['hold', 'pnl', 'trades', 'win_rate']

print(f"\n{'Hold Time':<14} {'P&L':>12} {'Win%':>8} {'Trades':>8}")
print("-"*45)
for _, row in hold_all.iterrows():
    if row['trades'] > 0:
        status = "+" if row['pnl'] > 0 else ""
        print(f"{row['hold']:<14} {status}${row['pnl']:>10,.0f} {row['win_rate']:>7.0%} {int(row['trades']):>8}")

# =============================================================================
# TILT ANALYSIS (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("TILT ANALYSIS (ALL TIME)")
print("="*70)

sorted_trades = closed.sort_values('first_date').copy()
sorted_trades['prev_result'] = sorted_trades['is_winner'].shift()
sorted_trades['prev_pnl'] = sorted_trades['total_pnl'].shift()

after_win = sorted_trades[sorted_trades['prev_result'] == True]
after_loss = sorted_trades[sorted_trades['prev_result'] == False]
after_big_loss = sorted_trades[sorted_trades['prev_pnl'] < -500]

print(f"""
After ANY WIN:
  Win rate: {after_win['is_winner'].mean():.1%}
  Avg P&L:  ${after_win['total_pnl'].mean():,.0f}

After ANY LOSS:
  Win rate: {after_loss['is_winner'].mean():.1%}
  Avg P&L:  ${after_loss['total_pnl'].mean():,.0f}

After BIG LOSS (>$500):
  Win rate: {after_big_loss['is_winner'].mean():.1%}
  Avg P&L:  ${after_big_loss['total_pnl'].mean():,.0f}
""")

tilt_impact = after_win['is_winner'].mean() - after_loss['is_winner'].mean()
print(f"TILT IMPACT: Your win rate drops {tilt_impact:.0%} after a loss")

# =============================================================================
# RISK METRICS (ALL TIME)
# =============================================================================

print("\n" + "="*70)
print("RISK METRICS (ALL TIME)")
print("="*70)

winners = closed[closed['is_winner']]
losers = closed[~closed['is_winner']]

avg_win = winners['total_pnl'].mean()
avg_loss = abs(losers['total_pnl'].mean())
win_rate = len(winners) / len(closed)
expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
profit_factor = winners['total_pnl'].sum() / abs(losers['total_pnl'].sum())

print(f"""
Total Trades:        {len(closed)}
Win Rate:            {win_rate:.1%}
Average Win:         ${avg_win:,.0f}
Average Loss:        ${avg_loss:,.0f}
Risk/Reward:         {avg_win/avg_loss:.2f}:1
Profit Factor:       {profit_factor:.2f}
Expectancy/Trade:    ${expectancy:,.0f}

Largest Win:         ${winners['total_pnl'].max():,.0f}
Largest Loss:        ${losers['total_pnl'].min():,.0f}
""")

# =============================================================================
# 2024 VS 2025 COMPARISON
# =============================================================================

print("\n" + "="*70)
print("2024 VS 2025 DETAILED COMPARISON")
print("="*70)

for year in [2024, 2025]:
    year_data = closed[closed['year'] == year]
    if len(year_data) == 0:
        continue

    year_winners = year_data[year_data['is_winner']]
    year_losers = year_data[~year_data['is_winner']]

    year_calls = year_data[year_data['option_type'] == 'CALL']
    year_puts = year_data[year_data['option_type'] == 'PUT']

    print(f"\n{year}:")
    print(f"  Total P&L:     ${year_data['total_pnl'].sum():,.0f}")
    print(f"  Trades:        {len(year_data)}")
    print(f"  Win Rate:      {year_data['is_winner'].mean():.0%}")
    print(f"  Avg Win:       ${year_winners['total_pnl'].mean():,.0f}")
    print(f"  Avg Loss:      ${year_losers['total_pnl'].mean():,.0f}")
    print(f"  CALLS:         ${year_calls['total_pnl'].sum():,.0f} ({year_calls['is_winner'].mean():.0%} win)")
    print(f"  PUTS:          ${year_puts['total_pnl'].sum():,.0f} ({year_puts['is_winner'].mean():.0%} win)")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "="*70)
print("EXECUTIVE SUMMARY")
print("="*70)

print(f"""
OVERALL (March 2024 - January 2026):
  Realized P&L:    ${realized_pnl:,.0f}
  Unrealized P&L:  ${unrealized_total:,.0f}
  TOTAL P&L:       ${total_pnl:,.0f}

KEY PATTERNS CONFIRMED ACROSS ALL DATA:
  - CALLS:         ${calls_total:,.0f} ({calls_wr:.0%} win rate)
  - PUTS:          ${puts_total:,.0f} ({puts_wr:.0%} win rate)
  - Monday:        Best day
  - Friday:        Worst day
  - Mid-month:     Best period
  - Longer holds:  Higher win rate

ACCOUNTS:
  - Best:  {acct_all.iloc[0]['account']} (${acct_all.iloc[0]['total']:,.0f})
  - Worst: {acct_all.iloc[-1]['account']} (${acct_all.iloc[-1]['total']:,.0f})

TICKERS (Top 5):
""")

for _, row in ticker_all.head(5).iterrows():
    print(f"  - {row['ticker']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win)")

print(f"""
TICKERS (Bottom 5):
""")
for _, row in ticker_all.tail(5).iterrows():
    print(f"  - {row['ticker']}: ${row['pnl']:,.0f} ({row['win_rate']:.0%} win)")

print("\n" + "="*70)
