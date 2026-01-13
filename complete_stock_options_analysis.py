"""
Complete Analysis: Stocks AND Options
"""
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES, POSITIONS_FILE
except ImportError:
    TRADE_FILES = ["./data/trades.csv"]
    POSITIONS_FILE = "./data/positions.csv"

all_trades = []
for f in TRADE_FILES:
    df = pd.read_csv(f, encoding='utf-8-sig')
    df = df[df['Run Date'].notna()]
    df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage|informational', case=False, na=False)]
    all_trades.append(df)

trades_df = pd.concat(all_trades, ignore_index=True)
trades_df = trades_df[trades_df['Action'].str.contains('YOU BOUGHT|YOU SOLD|ASSIGNED|EXPIRED', case=False, na=False)]
trades_df = trades_df[~trades_df['Action'].str.contains('DIVIDEND|REINVESTMENT', case=False, na=False)]

def parse_trade(row):
    action = str(row['Action'])
    symbol = str(row['Symbol'])

    is_call = 'CALL' in action
    is_put = 'PUT' in action

    if is_call:
        trade_type = 'CALL'
    elif is_put:
        trade_type = 'PUT'
    else:
        trade_type = 'STOCK'

    underlying = symbol
    if is_call or is_put:
        match = re.search(r'\(([A-Z]+)\)', action)
        if match: underlying = match.group(1)

    return pd.Series({'trade_type': trade_type, 'underlying': underlying})

parsed = trades_df.apply(parse_trade, axis=1)
trades_df = pd.concat([trades_df, parsed], axis=1)
trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')

# Build closed positions - INCLUDING stocks
symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'trade_type']).agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'Run Date': ['min', 'max'],
    'Action': 'count'
}).reset_index()
symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'trade_type', 'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions']

closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
closed['is_winner'] = closed['total_pnl'] > 0
closed['first_date'] = pd.to_datetime(closed['first_date'])
closed['last_date'] = pd.to_datetime(closed['last_date'])
closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days

print('='*70)
print('COMPLETE ANALYSIS: STOCKS + OPTIONS')
print('='*70)

# Overall breakdown
print('\n--- CLOSED POSITIONS BY TYPE ---')
for t in ['CALL', 'PUT', 'STOCK']:
    subset = closed[closed['trade_type'] == t]
    if len(subset) > 0:
        total = subset['total_pnl'].sum()
        wins = len(subset[subset['is_winner']])
        wr = wins/len(subset)*100
        median = subset['total_pnl'].median()
        print(f"  {t:>6}: {len(subset):>4} trades | ${total:>+10,.0f} total | {wr:>5.1f}% win rate | ${median:>+6,.0f} median")

print('\n--- OVERALL COMPARISON ---')
options = closed[closed['trade_type'].isin(['CALL', 'PUT'])]
stocks = closed[closed['trade_type'] == 'STOCK']

if len(options) > 0:
    print(f"  OPTIONS: {len(options)} closed | ${options['total_pnl'].sum():>+10,.0f} | {options['is_winner'].mean()*100:.1f}% win rate")
if len(stocks) > 0:
    print(f"  STOCKS:  {len(stocks)} closed | ${stocks['total_pnl'].sum():>+10,.0f} | {stocks['is_winner'].mean()*100:.1f}% win rate")

# Stock performance by ticker
if len(stocks) > 0:
    print('\n--- STOCK PERFORMANCE BY TICKER (closed positions) ---')
    stock_perf = stocks.groupby('underlying').agg({
        'total_pnl': ['sum', 'count', 'median'],
        'is_winner': 'mean'
    }).reset_index()
    stock_perf.columns = ['Ticker', 'Total', 'Trades', 'Median', 'WinRate']
    stock_perf = stock_perf.sort_values('Total', ascending=False)

    print(f"  {'Ticker':<8} | {'Total P&L':>12} | {'Trades':>6} | {'Median':>8} | {'Win%':>6}")
    print("  " + "-"*55)
    for _, r in stock_perf.iterrows():
        print(f"  {r['Ticker']:<8} | ${r['Total']:>+10,.0f} | {int(r['Trades']):>6} | ${r['Median']:>+6,.0f} | {r['WinRate']*100:>5.1f}%")

    # Compare key metrics
    print('\n--- HEAD TO HEAD: STOCKS vs OPTIONS ---')
    print(f"  {'Metric':<25} | {'Stocks':>12} | {'Options':>12}")
    print("  " + "-"*55)

    stock_winners_df = stocks[stocks['is_winner']]
    stock_losers_df = stocks[~stocks['is_winner']]
    opt_winners_df = options[options['is_winner']]
    opt_losers_df = options[~options['is_winner']]

    metrics = [
        ('Total P&L', stocks['total_pnl'].sum(), options['total_pnl'].sum()),
        ('Win Rate %', stocks['is_winner'].mean()*100, options['is_winner'].mean()*100),
        ('Median Trade', stocks['total_pnl'].median(), options['total_pnl'].median()),
        ('Avg Winner', stock_winners_df['total_pnl'].mean() if len(stock_winners_df) > 0 else 0, opt_winners_df['total_pnl'].mean() if len(opt_winners_df) > 0 else 0),
        ('Avg Loser', stock_losers_df['total_pnl'].mean() if len(stock_losers_df) > 0 else 0, opt_losers_df['total_pnl'].mean() if len(opt_losers_df) > 0 else 0),
        ('Avg Hold (days)', stocks['hold_days'].mean(), options['hold_days'].mean()),
    ]

    for name, stock_val, opt_val in metrics:
        if 'Rate' in name or '%' in name:
            print(f"  {name:<25} | {stock_val:>11.1f}% | {opt_val:>11.1f}%")
        elif 'Hold' in name or 'days' in name:
            print(f"  {name:<25} | {stock_val:>11.1f}d | {opt_val:>11.1f}d")
        else:
            print(f"  {name:<25} | ${stock_val:>+10,.0f} | ${opt_val:>+10,.0f}")

    # Profit factor
    stock_gross_profit = stock_winners_df['total_pnl'].sum() if len(stock_winners_df) > 0 else 0
    stock_gross_loss = abs(stock_losers_df['total_pnl'].sum()) if len(stock_losers_df) > 0 else 1
    opt_gross_profit = opt_winners_df['total_pnl'].sum() if len(opt_winners_df) > 0 else 0
    opt_gross_loss = abs(opt_losers_df['total_pnl'].sum()) if len(opt_losers_df) > 0 else 1

    print(f"  {'Profit Factor':<25} | {stock_gross_profit/stock_gross_loss:>12.2f} | {opt_gross_profit/opt_gross_loss:>12.2f}")

    # Best and worst stock trades
    print('\n--- BEST STOCK TRADES ---')
    for _, t in stocks.nlargest(5, 'total_pnl').iterrows():
        print(f"  ${t['total_pnl']:>+8,.0f} | {t['underlying']} | {t['first_date'].strftime('%Y-%m-%d')}")

    print('\n--- WORST STOCK TRADES ---')
    for _, t in stocks.nsmallest(5, 'total_pnl').iterrows():
        print(f"  ${t['total_pnl']:>+8,.0f} | {t['underlying']} | {t['first_date'].strftime('%Y-%m-%d')}")

# Combined analysis - all closed positions
print('\n' + '='*70)
print('REVISED TOTAL ANALYSIS (ALL INSTRUMENTS)')
print('='*70)

all_closed = closed.copy()
total_pnl = all_closed['total_pnl'].sum()
total_trades = len(all_closed)
win_rate = all_closed['is_winner'].mean() * 100
median_trade = all_closed['total_pnl'].median()

winners = all_closed[all_closed['is_winner']]
losers = all_closed[~all_closed['is_winner']]

avg_win = winners['total_pnl'].mean() if len(winners) > 0 else 0
avg_loss = losers['total_pnl'].mean() if len(losers) > 0 else 0
profit_factor = winners['total_pnl'].sum() / abs(losers['total_pnl'].sum()) if len(losers) > 0 else 0

print(f"""
  Total Closed Positions: {total_trades}
  Total P&L:              ${total_pnl:>+,.0f}
  Win Rate:               {win_rate:.1f}%
  Median Trade:           ${median_trade:>+,.0f}
  Avg Winner:             ${avg_win:>+,.0f}
  Avg Loser:              ${avg_loss:>+,.0f}
  Profit Factor:          {profit_factor:.2f}
""")

# Expectancy
payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0
expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
breakeven_wr = 1 / (1 + payoff_ratio) * 100 if payoff_ratio > 0 else 0

print(f"  Payoff Ratio:           {payoff_ratio:.2f}")
print(f"  Expectancy per trade:   ${expectancy:>+,.0f}")
print(f"  Breakeven win rate:     {breakeven_wr:.1f}%")

# Current open positions summary
print('\n' + '='*70)
print('YOUR CURRENT OPEN POSITIONS (from positions file)')
print('='*70)

positions_file = POSITIONS_FILE
try:
    pos_df = pd.read_csv(positions_file, encoding='utf-8-sig', index_col=False)

    # Filter to actual positions (not cash)
    pos_df = pos_df[pos_df['Symbol'].notna()]
    pos_df = pos_df[~pos_df['Symbol'].str.contains('SPAXX|FDRXX|FCASH|USD|Pending', case=False, na=False)]

    # Clean dollar values
    def clean_dollar(val):
        if pd.isna(val): return 0
        val = str(val).replace('$', '').replace(',', '').replace('+', '')
        try: return float(val)
        except: return 0

    pos_df['Unrealized'] = pos_df['Total Gain/Loss Dollar'].apply(clean_dollar)
    pos_df['Value'] = pos_df['Current Value'].apply(clean_dollar)

    # Identify type
    pos_df['Type'] = pos_df['Symbol'].apply(lambda x: 'OPTION' if str(x).startswith(' -') else 'STOCK')

    stock_positions = pos_df[pos_df['Type'] == 'STOCK']
    option_positions = pos_df[pos_df['Type'] == 'OPTION']

    print(f"\n  STOCK POSITIONS: {len(stock_positions)}")
    print(f"  {'Symbol':<8} | {'Value':>12} | {'Unrealized':>12}")
    print("  " + "-"*40)
    for _, p in stock_positions.iterrows():
        print(f"  {str(p['Symbol'])[:8]:<8} | ${p['Value']:>10,.0f} | ${p['Unrealized']:>+10,.0f}")

    print(f"\n  Total Stock Value:      ${stock_positions['Value'].sum():>,.0f}")
    print(f"  Total Stock Unrealized: ${stock_positions['Unrealized'].sum():>+,.0f}")

    print(f"\n  OPTION POSITIONS: {len(option_positions)}")
    print(f"  {'Symbol':<25} | {'Value':>10} | {'Unrealized':>12}")
    print("  " + "-"*55)
    for _, p in option_positions.iterrows():
        sym = str(p['Symbol']).strip()[:25]
        print(f"  {sym:<25} | ${p['Value']:>8,.0f} | ${p['Unrealized']:>+10,.0f}")

    print(f"\n  Total Option Value:      ${option_positions['Value'].sum():>,.0f}")
    print(f"  Total Option Unrealized: ${option_positions['Unrealized'].sum():>+,.0f}")

    print(f"\n  COMBINED UNREALIZED:     ${pos_df['Unrealized'].sum():>+,.0f}")

except Exception as e:
    print(f"  Error reading positions: {e}")

print('\n' + '='*70)
