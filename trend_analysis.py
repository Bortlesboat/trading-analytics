"""
Deep Trend Analysis - Finding Patterns and Winning Strategies
Uses the same trade processing as full_history_analysis.py
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import warnings
warnings.filterwarnings('ignore')

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES
except ImportError:
    TRADE_FILES = ["./data/trades.csv"]

def clean_dollar(val):
    """Convert dollar strings to float"""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).replace('$', '').replace(',', '').replace('+', '').strip()
    if val == '' or val == '--':
        return 0.0
    try:
        return float(val)
    except:
        return 0.0

def load_and_process_trades():
    """Load all trades and build closed positions"""
    # Load all files
    all_trades = []
    for f in TRADE_FILES:
        try:
            df = pd.read_csv(f, encoding='utf-8-sig')
            df = df[df['Run Date'].notna()]
            df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage|informational', case=False, na=False)]
            all_trades.append(df)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    trades_df = pd.concat(all_trades, ignore_index=True)

    # Filter to actual trades
    trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
    exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
    trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)]
    trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

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
    trades_df['week_of_year'] = trades_df['Run Date'].dt.isocalendar().week
    trades_df['year_week'] = trades_df['Run Date'].dt.strftime('%Y-W%U')
    trades_df['year_month'] = trades_df['Run Date'].dt.to_period('M')

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

    # Build closed positions by grouping
    symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type', 'account_type']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',
        'Run Date': ['min', 'max'],
        'direction': 'count',
        'year': 'first',
        'month': 'first',
        'day_of_week': 'first',
        'day_of_month': 'first',
        'week_of_year': 'first',
        'year_week': 'first',
        'year_month': 'first'
    }).reset_index()
    symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type', 'account_type',
                           'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions',
                           'year', 'month', 'day_of_week', 'day_of_month', 'week_of_year', 'year_week', 'year_month']

    # Closed positions: net quantity near 0, at least 2 transactions
    closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
    closed['is_winner'] = closed['total_pnl'] > 0
    closed['first_date'] = pd.to_datetime(closed['first_date'])
    closed['last_date'] = pd.to_datetime(closed['last_date'])
    closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days

    # Calculate position size (estimate from P&L and typical option returns)
    closed['est_position_size'] = closed['total_pnl'].abs() * 2  # rough estimate

    return trades_df, closed

def print_section(title):
    print(f"\n{'='*70}")
    print(title)
    print('='*70)

def analyze_streaks(df):
    """Find winning and losing streaks"""
    print_section("STREAK ANALYSIS")

    df_sorted = df.sort_values('last_date').reset_index(drop=True)

    # Calculate streaks
    streaks = []
    current_type = None
    current_start = 0
    current_pnl = 0

    for i, row in df_sorted.iterrows():
        if current_type is None:
            current_type = row['is_winner']
            current_start = i
            current_pnl = row['total_pnl']
        elif row['is_winner'] == current_type:
            current_pnl += row['total_pnl']
        else:
            streaks.append({
                'Type': 'Win' if current_type else 'Loss',
                'Length': i - current_start,
                'StartDate': df_sorted.loc[current_start, 'last_date'],
                'EndDate': df_sorted.loc[i-1, 'last_date'],
                'PnL': current_pnl
            })
            current_type = row['is_winner']
            current_start = i
            current_pnl = row['total_pnl']

    # Add final streak
    if len(df_sorted) > 0:
        streaks.append({
            'Type': 'Win' if current_type else 'Loss',
            'Length': len(df_sorted) - current_start,
            'StartDate': df_sorted.loc[current_start, 'last_date'],
            'EndDate': df_sorted.iloc[-1]['last_date'],
            'PnL': current_pnl
        })

    streak_df = pd.DataFrame(streaks)

    # Best winning streaks
    win_streaks = streak_df[streak_df['Type'] == 'Win'].nlargest(10, 'Length')
    print("\nTOP 10 WINNING STREAKS:")
    print("-" * 65)
    for _, s in win_streaks.iterrows():
        print(f"  {s['Length']:>2} wins | {s['StartDate'].strftime('%Y-%m-%d')} to {s['EndDate'].strftime('%Y-%m-%d')} | ${s['PnL']:>+9,.0f}")

    # Worst losing streaks
    loss_streaks = streak_df[streak_df['Type'] == 'Loss'].nlargest(10, 'Length')
    print("\nTOP 10 LOSING STREAKS:")
    print("-" * 65)
    for _, s in loss_streaks.iterrows():
        print(f"  {s['Length']:>2} losses | {s['StartDate'].strftime('%Y-%m-%d')} to {s['EndDate'].strftime('%Y-%m-%d')} | ${s['PnL']:>+9,.0f}")

    # Most profitable stretches
    print("\nMOST PROFITABLE WINNING STRETCHES:")
    print("-" * 65)
    top_profit = streak_df[streak_df['Type'] == 'Win'].nlargest(10, 'PnL')
    for _, s in top_profit.iterrows():
        print(f"  ${s['PnL']:>+9,.0f} | {s['Length']:>2} wins | {s['StartDate'].strftime('%Y-%m-%d')} to {s['EndDate'].strftime('%Y-%m-%d')}")

    # Worst drawdowns
    print("\nWORST LOSING STRETCHES:")
    print("-" * 65)
    worst_loss = streak_df[streak_df['Type'] == 'Loss'].nsmallest(10, 'PnL')
    for _, s in worst_loss.iterrows():
        print(f"  ${s['PnL']:>+9,.0f} | {s['Length']:>2} losses | {s['StartDate'].strftime('%Y-%m-%d')} to {s['EndDate'].strftime('%Y-%m-%d')}")

    return streak_df

def analyze_rolling_performance(df):
    """Analyze rolling windows of performance"""
    print_section("ROLLING PERFORMANCE ANALYSIS")

    # Weekly P&L using last_date
    df['close_year_week'] = df['last_date'].dt.strftime('%Y-W%U')
    weekly = df.groupby('close_year_week').agg({
        'total_pnl': 'sum',
        'is_winner': ['sum', 'count']
    }).reset_index()
    weekly.columns = ['Week', 'PnL', 'Wins', 'Total']
    weekly['WinRate'] = (weekly['Wins'] / weekly['Total'] * 100).round(0)

    # Best weeks
    print("\nTOP 15 BEST WEEKS:")
    print("-" * 70)
    best_weeks = weekly.nlargest(15, 'PnL')
    for _, w in best_weeks.iterrows():
        print(f"  {w['Week']}: ${w['PnL']:>+10,.0f} | {w['WinRate']:>3.0f}% win rate | {int(w['Total']):>2} trades")

    # Worst weeks
    print("\nTOP 15 WORST WEEKS:")
    print("-" * 70)
    worst_weeks = weekly.nsmallest(15, 'PnL')
    for _, w in worst_weeks.iterrows():
        print(f"  {w['Week']}: ${w['PnL']:>+10,.0f} | {w['WinRate']:>3.0f}% win rate | {int(w['Total']):>2} trades")

    # Monthly analysis using close date
    df['close_year_month'] = df['last_date'].dt.to_period('M')
    monthly = df.groupby('close_year_month').agg({
        'total_pnl': 'sum',
        'is_winner': ['sum', 'count']
    }).reset_index()
    monthly.columns = ['Month', 'PnL', 'Wins', 'Total']
    monthly['WinRate'] = (monthly['Wins'] / monthly['Total'] * 100).round(0)

    print("\nMONTHLY P&L TREND (ALL MONTHS):")
    print("-" * 70)
    for _, m in monthly.iterrows():
        bar_len = min(int(abs(m['PnL']) / 2000), 25)
        bar = '+' * bar_len if m['PnL'] > 0 else '-' * bar_len
        print(f"  {m['Month']}: ${m['PnL']:>+10,.0f} | {m['WinRate']:>3.0f}% | {bar}")

    return weekly, monthly

def analyze_combinations(df):
    """Analyze winning combinations of factors"""
    print_section("WINNING COMBINATIONS")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Use first date (entry date) for day analysis
    df['entry_day'] = df['first_date'].dt.dayofweek
    df['entry_day_name'] = df['entry_day'].apply(lambda x: day_names[int(x)] if pd.notna(x) else 'Unknown')

    # Day + Type combination
    print("\nDAY OF ENTRY + OPTION TYPE:")
    print("-" * 70)
    combo1 = df.groupby(['entry_day_name', 'option_type']).agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    combo1.columns = ['Day', 'Type', 'PnL', 'WinRate', 'Count']
    combo1 = combo1[combo1['Count'] >= 10].sort_values('PnL', ascending=False)

    for _, c in combo1.iterrows():
        if pd.notna(c['Type']):
            print(f"  {c['Day']:>3} {c['Type']:<4}: ${c['PnL']:>+10,.0f} | {c['WinRate']*100:>5.1f}% win | {int(c['Count']):>3} trades")

    # Ticker + Type combination
    print("\nTICKER + OPTION TYPE (min 10 trades):")
    print("-" * 70)
    combo2 = df.groupby(['underlying', 'option_type']).agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    combo2.columns = ['Ticker', 'Type', 'PnL', 'WinRate', 'Count']
    combo2 = combo2[(combo2['Count'] >= 10) & (combo2['Type'].notna())].sort_values('PnL', ascending=False)

    print("\n  BEST TICKER+TYPE COMBOS:")
    for _, c in combo2.head(12).iterrows():
        print(f"    {c['Ticker']:>5} {c['Type']:<4}: ${c['PnL']:>+10,.0f} | {c['WinRate']*100:>5.1f}% win | {int(c['Count']):>3} trades")

    print("\n  WORST TICKER+TYPE COMBOS:")
    for _, c in combo2.tail(12).iterrows():
        print(f"    {c['Ticker']:>5} {c['Type']:<4}: ${c['PnL']:>+10,.0f} | {c['WinRate']*100:>5.1f}% win | {int(c['Count']):>3} trades")

    # Hold time + Type
    print("\nHOLD TIME + OPTION TYPE:")
    print("-" * 70)

    def hold_bucket(days):
        if pd.isna(days): return '?: Unknown'
        if days == 0: return '0: Same day'
        elif days == 1: return '1: 1 day'
        elif days <= 3: return '2: 2-3 days'
        elif days <= 7: return '3: 4-7 days'
        elif days <= 14: return '4: 1-2 weeks'
        elif days <= 30: return '5: 2-4 weeks'
        else: return '6: 1+ month'

    df['hold_bucket'] = df['hold_days'].apply(hold_bucket)

    combo3 = df.groupby(['hold_bucket', 'option_type']).agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    combo3.columns = ['Hold', 'Type', 'PnL', 'WinRate', 'Count']
    combo3 = combo3[(combo3['Count'] >= 5) & (combo3['Type'].notna())].sort_values('PnL', ascending=False)

    for _, c in combo3.iterrows():
        hold_label = c['Hold'].split(': ')[1] if ': ' in str(c['Hold']) else c['Hold']
        print(f"  {hold_label:>10} {c['Type']:<4}: ${c['PnL']:>+10,.0f} | {c['WinRate']*100:>5.1f}% win | {int(c['Count']):>3} trades")

    # Account + Type
    print("\nACCOUNT + OPTION TYPE:")
    print("-" * 70)
    combo4 = df.groupby(['account_type', 'option_type']).agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    combo4.columns = ['Account', 'Type', 'PnL', 'WinRate', 'Count']
    combo4 = combo4[(combo4['Count'] >= 5) & (combo4['Type'].notna())].sort_values('PnL', ascending=False)

    for _, c in combo4.iterrows():
        print(f"  {c['Account']:>15} {c['Type']:<4}: ${c['PnL']:>+10,.0f} | {c['WinRate']*100:>5.1f}% win | {int(c['Count']):>3} trades")

def analyze_seasonal_patterns(df):
    """Analyze seasonal and calendar patterns"""
    print_section("CALENDAR PATTERNS")

    # Use close date for calendar analysis
    df['close_week_of_year'] = df['last_date'].dt.isocalendar().week
    df['close_day_of_month'] = df['last_date'].dt.day
    df['close_month'] = df['last_date'].dt.month

    # Week of year performance
    week_perf = df.groupby('close_week_of_year').agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    week_perf.columns = ['Week', 'PnL', 'WinRate', 'Count']
    week_perf = week_perf[week_perf['Count'] >= 5]

    print("\nBEST WEEKS OF YEAR (across all years):")
    print("-" * 55)
    for _, w in week_perf.nlargest(10, 'PnL').iterrows():
        print(f"  Week {int(w['Week']):>2}: ${w['PnL']:>+10,.0f} | {w['WinRate']*100:>5.1f}% win | {int(w['Count'])} trades")

    print("\nWORST WEEKS OF YEAR (across all years):")
    print("-" * 55)
    for _, w in week_perf.nsmallest(10, 'PnL').iterrows():
        print(f"  Week {int(w['Week']):>2}: ${w['PnL']:>+10,.0f} | {w['WinRate']*100:>5.1f}% win | {int(w['Count'])} trades")

    # Day of month patterns
    dom_perf = df.groupby('close_day_of_month').agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    dom_perf.columns = ['Day', 'PnL', 'WinRate', 'Count']

    print("\nBEST DAYS OF MONTH:")
    print("-" * 55)
    for _, d in dom_perf.nlargest(10, 'PnL').iterrows():
        print(f"  Day {int(d['Day']):>2}: ${d['PnL']:>+10,.0f} | {d['WinRate']*100:>5.1f}% win | {int(d['Count'])} trades")

    print("\nWORST DAYS OF MONTH:")
    print("-" * 55)
    for _, d in dom_perf.nsmallest(10, 'PnL').iterrows():
        print(f"  Day {int(d['Day']):>2}: ${d['PnL']:>+10,.0f} | {d['WinRate']*100:>5.1f}% win | {int(d['Count'])} trades")

    # Month name performance
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    month_perf = df.groupby('close_month').agg({
        'total_pnl': 'sum',
        'is_winner': ['mean', 'count']
    }).reset_index()
    month_perf.columns = ['Month', 'PnL', 'WinRate', 'Count']

    print("\nPERFORMANCE BY MONTH (ALL YEARS COMBINED):")
    print("-" * 55)
    for _, m in month_perf.sort_values('Month').iterrows():
        name = month_names[int(m['Month'])]
        bar = '+' * min(int(m['PnL']/2000), 15) if m['PnL'] > 0 else '-' * min(int(abs(m['PnL'])/2000), 15)
        print(f"  {name}: ${m['PnL']:>+10,.0f} | {m['WinRate']*100:>5.1f}% | {bar}")

def analyze_recovery_patterns(df):
    """Analyze how you recover from losses"""
    print_section("RECOVERY ANALYSIS")

    df_sorted = df.sort_values('last_date').reset_index(drop=True)

    # Calculate cumulative P&L
    df_sorted['CumPnL'] = df_sorted['total_pnl'].cumsum()
    df_sorted['RunningMax'] = df_sorted['CumPnL'].cummax()
    df_sorted['Drawdown'] = df_sorted['CumPnL'] - df_sorted['RunningMax']

    # Find major drawdown periods
    print("\nMAJOR DRAWDOWN PERIODS (>$5K):")
    print("-" * 75)

    in_drawdown = False
    drawdown_start = None
    drawdown_start_idx = None
    drawdown_bottom = 0
    drawdown_bottom_date = None
    drawdowns = []

    for i, row in df_sorted.iterrows():
        if row['Drawdown'] < 0 and not in_drawdown:
            in_drawdown = True
            drawdown_start_idx = i
            drawdown_start = row['last_date']
            drawdown_bottom = row['Drawdown']
            drawdown_bottom_date = row['last_date']
        elif in_drawdown:
            if row['Drawdown'] < drawdown_bottom:
                drawdown_bottom = row['Drawdown']
                drawdown_bottom_date = row['last_date']
            if row['Drawdown'] == 0:
                in_drawdown = False
                if drawdown_bottom < -5000:
                    drawdowns.append({
                        'Start': drawdown_start,
                        'Bottom': drawdown_bottom_date,
                        'End': row['last_date'],
                        'MaxDrawdown': drawdown_bottom,
                        'Duration': (row['last_date'] - drawdown_start).days,
                        'TradesToRecover': i - drawdown_start_idx
                    })

    # If still in drawdown at end
    if in_drawdown and drawdown_bottom < -5000:
        drawdowns.append({
            'Start': drawdown_start,
            'Bottom': drawdown_bottom_date,
            'End': None,
            'MaxDrawdown': drawdown_bottom,
            'Duration': None,
            'TradesToRecover': None
        })

    for dd in sorted(drawdowns, key=lambda x: x['MaxDrawdown'])[:10]:
        end_str = dd['End'].strftime('%Y-%m-%d') if dd['End'] else 'ONGOING'
        dur_str = f"{dd['Duration']} days" if dd['Duration'] else 'N/A'
        trades_str = f"{dd['TradesToRecover']} trades" if dd['TradesToRecover'] else 'N/A'
        print(f"  ${dd['MaxDrawdown']:>+10,.0f} | {dd['Start'].strftime('%Y-%m-%d')} to {end_str} | {dur_str} | {trades_str}")

    # What happens after big losses?
    print("\nBEHAVIOR AFTER BIG LOSSES (>$1K):")
    print("-" * 75)

    big_loss_indices = df_sorted[df_sorted['total_pnl'] < -1000].index

    next_trade_outcomes = []
    next_5_outcomes = []

    for idx in big_loss_indices:
        if idx + 1 < len(df_sorted):
            next_trade_outcomes.append({
                'win': df_sorted.loc[idx + 1, 'is_winner'],
                'pnl': df_sorted.loc[idx + 1, 'total_pnl']
            })
        if idx + 5 < len(df_sorted):
            next_5 = df_sorted.loc[idx+1:idx+5]
            next_5_outcomes.append({
                'wins': next_5['is_winner'].sum(),
                'pnl': next_5['total_pnl'].sum()
            })

    if next_trade_outcomes:
        win_rate = sum(1 for x in next_trade_outcomes if x['win']) / len(next_trade_outcomes) * 100
        avg_pnl = sum(x['pnl'] for x in next_trade_outcomes) / len(next_trade_outcomes)
        print(f"  Immediate next trade: {win_rate:.1f}% win rate, ${avg_pnl:,.0f} avg P&L")

    if next_5_outcomes:
        avg_wins = sum(x['wins'] for x in next_5_outcomes) / len(next_5_outcomes)
        avg_pnl = sum(x['pnl'] for x in next_5_outcomes) / len(next_5_outcomes)
        print(f"  Next 5 trades: {avg_wins:.1f} avg wins, ${avg_pnl:,.0f} avg P&L")

    # Compare to baseline
    baseline_win_rate = df_sorted['is_winner'].mean() * 100
    print(f"\n  Baseline win rate: {baseline_win_rate:.1f}%")

def analyze_best_strategies(df):
    """Find the best performing strategy combinations"""
    print_section("STRATEGY FINDER - WHAT ACTUALLY WORKS")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    # Create comprehensive strategy features
    df = df.copy()
    df['entry_day'] = df['first_date'].dt.dayofweek
    df['entry_day_name'] = df['entry_day'].apply(lambda x: day_names[int(x)] if pd.notna(x) else 'Unknown')

    def month_period(day):
        if pd.isna(day): return 'Unknown'
        if day <= 10: return 'Early'
        elif day <= 20: return 'Mid'
        else: return 'Late'

    df['month_period'] = df['first_date'].dt.day.apply(month_period)

    def hold_cat(days):
        if pd.isna(days): return 'Unknown'
        if days <= 1: return 'Quick'
        elif days <= 7: return 'Week'
        else: return 'Swing'

    df['hold_cat'] = df['hold_days'].apply(hold_cat)

    # Test various strategy combinations
    strategies = []

    # Single factor: Day of week
    for day in day_names[:5]:
        subset = df[df['entry_day_name'] == day]
        if len(subset) >= 20:
            strategies.append({
                'Strategy': f'{day} entries',
                'PnL': subset['total_pnl'].sum(),
                'WinRate': subset['is_winner'].mean() * 100,
                'Trades': len(subset),
                'AvgPnL': subset['total_pnl'].mean()
            })

    # Single factor: Option type
    for opt_type in ['PUT', 'CALL']:
        subset = df[df['option_type'] == opt_type]
        if len(subset) >= 20:
            strategies.append({
                'Strategy': f'{opt_type}s only',
                'PnL': subset['total_pnl'].sum(),
                'WinRate': subset['is_winner'].mean() * 100,
                'Trades': len(subset),
                'AvgPnL': subset['total_pnl'].mean()
            })

    # Two factor: Day + Type
    for day in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']:
        for opt_type in ['PUT', 'CALL']:
            subset = df[(df['entry_day_name'] == day) & (df['option_type'] == opt_type)]
            if len(subset) >= 10:
                strategies.append({
                    'Strategy': f'{day} {opt_type}s',
                    'PnL': subset['total_pnl'].sum(),
                    'WinRate': subset['is_winner'].mean() * 100,
                    'Trades': len(subset),
                    'AvgPnL': subset['total_pnl'].mean()
                })

    # Two factor: Month period + Type
    for period in ['Early', 'Mid', 'Late']:
        for opt_type in ['PUT', 'CALL']:
            subset = df[(df['month_period'] == period) & (df['option_type'] == opt_type)]
            if len(subset) >= 20:
                strategies.append({
                    'Strategy': f'{period}-month {opt_type}s',
                    'PnL': subset['total_pnl'].sum(),
                    'WinRate': subset['is_winner'].mean() * 100,
                    'Trades': len(subset),
                    'AvgPnL': subset['total_pnl'].mean()
                })

    # Two factor: Hold time + Type
    for hold in ['Quick', 'Week', 'Swing']:
        for opt_type in ['PUT', 'CALL']:
            subset = df[(df['hold_cat'] == hold) & (df['option_type'] == opt_type)]
            if len(subset) >= 20:
                strategies.append({
                    'Strategy': f'{hold}-hold {opt_type}s',
                    'PnL': subset['total_pnl'].sum(),
                    'WinRate': subset['is_winner'].mean() * 100,
                    'Trades': len(subset),
                    'AvgPnL': subset['total_pnl'].mean()
                })

    # Three factor combinations
    for day in ['Mon', 'Tue']:
        for period in ['Mid', 'Late']:
            subset = df[(df['entry_day_name'] == day) & (df['month_period'] == period) & (df['option_type'] == 'PUT')]
            if len(subset) >= 5:
                strategies.append({
                    'Strategy': f'{day} {period}-month PUTs',
                    'PnL': subset['total_pnl'].sum(),
                    'WinRate': subset['is_winner'].mean() * 100,
                    'Trades': len(subset),
                    'AvgPnL': subset['total_pnl'].mean()
                })

    # Whitelist tickers
    whitelist = ['ETHA', 'NVDA', 'AAPL', 'SPY', 'TSLA']
    subset = df[df['underlying'].isin(whitelist)]
    if len(subset) >= 20:
        strategies.append({
            'Strategy': 'Whitelist only',
            'PnL': subset['total_pnl'].sum(),
            'WinRate': subset['is_winner'].mean() * 100,
            'Trades': len(subset),
            'AvgPnL': subset['total_pnl'].mean()
        })

        for opt_type in ['PUT', 'CALL']:
            type_subset = subset[subset['option_type'] == opt_type]
            if len(type_subset) >= 10:
                strategies.append({
                    'Strategy': f'Whitelist {opt_type}s',
                    'PnL': type_subset['total_pnl'].sum(),
                    'WinRate': type_subset['is_winner'].mean() * 100,
                    'Trades': len(type_subset),
                    'AvgPnL': type_subset['total_pnl'].mean()
                })

    # Avoid blacklist
    blacklist = ['MSTR', 'AMZN', 'SMCI', 'IBIT', 'SBET', 'GOOG']
    subset = df[~df['underlying'].isin(blacklist)]
    if len(subset) >= 20:
        strategies.append({
            'Strategy': 'Avoid blacklist',
            'PnL': subset['total_pnl'].sum(),
            'WinRate': subset['is_winner'].mean() * 100,
            'Trades': len(subset),
            'AvgPnL': subset['total_pnl'].mean()
        })

        for opt_type in ['PUT', 'CALL']:
            type_subset = subset[subset['option_type'] == opt_type]
            if len(type_subset) >= 10:
                strategies.append({
                    'Strategy': f'No blacklist {opt_type}s',
                    'PnL': type_subset['total_pnl'].sum(),
                    'WinRate': type_subset['is_winner'].mean() * 100,
                    'Trades': len(type_subset),
                    'AvgPnL': type_subset['total_pnl'].mean()
                })

    # Long hold strategies
    long_holds = df[df['hold_days'] >= 14]
    if len(long_holds) >= 10:
        strategies.append({
            'Strategy': '2+ week holds',
            'PnL': long_holds['total_pnl'].sum(),
            'WinRate': long_holds['is_winner'].mean() * 100,
            'Trades': len(long_holds),
            'AvgPnL': long_holds['total_pnl'].mean()
        })

    very_long = df[df['hold_days'] >= 30]
    if len(very_long) >= 5:
        strategies.append({
            'Strategy': '1+ month holds',
            'PnL': very_long['total_pnl'].sum(),
            'WinRate': very_long['is_winner'].mean() * 100,
            'Trades': len(very_long),
            'AvgPnL': very_long['total_pnl'].mean()
        })

    # Sort and display
    strat_df = pd.DataFrame(strategies)

    if len(strat_df) > 0:
        print("\nTOP 20 STRATEGIES BY TOTAL P&L:")
        print("-" * 85)
        print(f"  {'Strategy':<30} | {'Total P&L':>12} | {'Win%':>6} | {'Trades':>6} | {'Avg P&L':>10}")
        print("-" * 85)
        for _, s in strat_df.nlargest(20, 'PnL').iterrows():
            print(f"  {s['Strategy']:<30} | ${s['PnL']:>+10,.0f} | {s['WinRate']:>5.1f}% | {int(s['Trades']):>6} | ${s['AvgPnL']:>+8,.0f}")

        print("\nTOP 15 STRATEGIES BY WIN RATE (min 20 trades):")
        print("-" * 85)
        high_count = strat_df[strat_df['Trades'] >= 20].nlargest(15, 'WinRate')
        for _, s in high_count.iterrows():
            print(f"  {s['Strategy']:<30} | ${s['PnL']:>+10,.0f} | {s['WinRate']:>5.1f}% | {int(s['Trades']):>6} | ${s['AvgPnL']:>+8,.0f}")

        print("\nWORST 15 STRATEGIES (AVOID THESE):")
        print("-" * 85)
        for _, s in strat_df.nsmallest(15, 'PnL').iterrows():
            print(f"  {s['Strategy']:<30} | ${s['PnL']:>+10,.0f} | {s['WinRate']:>5.1f}% | {int(s['Trades']):>6} | ${s['AvgPnL']:>+8,.0f}")

def analyze_trade_sequences(df):
    """Analyze sequences of trades"""
    print_section("TRADE SEQUENCE PATTERNS")

    df_sorted = df.sort_values('last_date').reset_index(drop=True)

    # What type of trade follows a winner vs loser?
    print("\nWHAT HAPPENS AFTER DIFFERENT OUTCOMES:")
    print("-" * 65)

    sequences = []
    for i in range(len(df_sorted) - 1):
        curr = df_sorted.iloc[i]
        next_trade = df_sorted.iloc[i + 1]
        if pd.notna(curr['option_type']) and pd.notna(next_trade['option_type']):
            sequences.append({
                'PrevWin': curr['is_winner'],
                'PrevPnL': curr['total_pnl'],
                'PrevType': curr['option_type'],
                'NextWin': next_trade['is_winner'],
                'NextPnL': next_trade['total_pnl'],
                'NextType': next_trade['option_type'],
            })

    if sequences:
        seq_df = pd.DataFrame(sequences)

        # After winning CALL
        after_win_call = seq_df[(seq_df['PrevWin']) & (seq_df['PrevType'] == 'CALL')]
        if len(after_win_call) > 10:
            print(f"  After winning CALL ({len(after_win_call)} instances):")
            print(f"    Next trade win rate: {after_win_call['NextWin'].mean()*100:.1f}%")
            if len(after_win_call['NextType'].mode()) > 0:
                print(f"    Usually trades: {after_win_call['NextType'].mode().iloc[0]}")

        # After losing CALL
        after_loss_call = seq_df[(~seq_df['PrevWin']) & (seq_df['PrevType'] == 'CALL')]
        if len(after_loss_call) > 10:
            print(f"  After losing CALL ({len(after_loss_call)} instances):")
            print(f"    Next trade win rate: {after_loss_call['NextWin'].mean()*100:.1f}%")

        # After winning PUT
        after_win_put = seq_df[(seq_df['PrevWin']) & (seq_df['PrevType'] == 'PUT')]
        if len(after_win_put) > 10:
            print(f"  After winning PUT ({len(after_win_put)} instances):")
            print(f"    Next trade win rate: {after_win_put['NextWin'].mean()*100:.1f}%")

        # After losing PUT
        after_loss_put = seq_df[(~seq_df['PrevWin']) & (seq_df['PrevType'] == 'PUT')]
        if len(after_loss_put) > 10:
            print(f"  After losing PUT ({len(after_loss_put)} instances):")
            print(f"    Next trade win rate: {after_loss_put['NextWin'].mean()*100:.1f}%")

    # Same ticker repeat trades
    print("\nREPEAT TICKER ANALYSIS:")
    print("-" * 65)

    df_sorted['PrevTicker'] = df_sorted['underlying'].shift(1)
    df_sorted['SameTicker'] = df_sorted['underlying'] == df_sorted['PrevTicker']

    same_ticker = df_sorted[df_sorted['SameTicker']]
    diff_ticker = df_sorted[~df_sorted['SameTicker']]

    if len(same_ticker) > 5:
        print(f"  Trading same ticker again: {same_ticker['is_winner'].mean()*100:.1f}% win rate ({len(same_ticker)} trades)")
    if len(diff_ticker) > 5:
        print(f"  Switching tickers:         {diff_ticker['is_winner'].mean()*100:.1f}% win rate ({len(diff_ticker)} trades)")

def analyze_year_over_year(df):
    """Compare performance year over year in detail"""
    print_section("YEAR OVER YEAR DEEP DIVE")

    df['close_year'] = df['last_date'].dt.year

    years = sorted(df['close_year'].unique())

    for year in years:
        year_df = df[df['close_year'] == year]
        print(f"\n{year} DETAILED BREAKDOWN:")
        print("-" * 65)

        # Monthly breakdown for this year
        year_df['close_month'] = year_df['last_date'].dt.month
        month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        monthly = year_df.groupby('close_month').agg({
            'total_pnl': 'sum',
            'is_winner': ['mean', 'count']
        }).reset_index()
        monthly.columns = ['Month', 'PnL', 'WinRate', 'Trades']

        print(f"  {'Month':<6} | {'P&L':>12} | {'Win%':>6} | {'Trades':>6}")
        print("  " + "-" * 45)
        for _, m in monthly.sort_values('Month').iterrows():
            name = month_names[int(m['Month'])]
            print(f"  {name:<6} | ${m['PnL']:>+10,.0f} | {m['WinRate']*100:>5.1f}% | {int(m['Trades']):>6}")

        # Best and worst months
        if len(monthly) >= 3:
            best = monthly.nlargest(1, 'PnL').iloc[0]
            worst = monthly.nsmallest(1, 'PnL').iloc[0]
            print(f"\n  Best month:  {month_names[int(best['Month'])]} (${best['PnL']:+,.0f})")
            print(f"  Worst month: {month_names[int(worst['Month'])]} (${worst['PnL']:+,.0f})")

def main():
    print("="*70)
    print("DEEP TREND ANALYSIS")
    print("Finding Patterns and Winning Strategies")
    print("="*70)

    # Load and process data
    print("\nLoading all trading data...")
    trades_df, closed = load_and_process_trades()

    print(f"Analyzed {len(closed)} closed positions")
    print(f"Date range: {closed['last_date'].min().strftime('%Y-%m-%d')} to {closed['last_date'].max().strftime('%Y-%m-%d')}")
    print(f"Total P&L: ${closed['total_pnl'].sum():,.0f}")

    # Run all analyses
    analyze_streaks(closed)
    analyze_rolling_performance(closed)
    analyze_combinations(closed)
    analyze_seasonal_patterns(closed)
    analyze_recovery_patterns(closed)
    analyze_trade_sequences(closed)
    analyze_year_over_year(closed)
    analyze_best_strategies(closed)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
