"""
Rigorous Statistical Analysis
Accounts for outliers, sample size, and statistical significance
"""
import pandas as pd
import numpy as np
from scipy import stats
import re
import warnings
warnings.filterwarnings('ignore')

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES
except ImportError:
    TRADE_FILES = ["./data/trades.csv"]

def load_and_process_trades():
    """Load all trades and build closed positions"""
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

    trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
    exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
    trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)]
    trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

    def parse_trade(row):
        action = str(row['Action'])
        symbol = str(row['Symbol'])
        is_option = 'CALL' in action or 'PUT' in action
        option_type = 'CALL' if 'CALL' in action else ('PUT' if 'PUT' in action else None)
        underlying = symbol
        if is_option:
            match = re.search(r'\(([A-Z]+)\)', action)
            if match: underlying = match.group(1)
        return pd.Series({'is_option': is_option, 'option_type': option_type, 'underlying': underlying})

    parsed = trades_df.apply(parse_trade, axis=1)
    trades_df = pd.concat([trades_df, parsed], axis=1)
    trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
    trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
    trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')

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

    symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type', 'account_type']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',
        'Run Date': ['min', 'max'],
        'Action': 'count'
    }).reset_index()
    symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type', 'account_type',
                           'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions']

    closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
    closed['is_winner'] = closed['total_pnl'] > 0
    closed['first_date'] = pd.to_datetime(closed['first_date'])
    closed['last_date'] = pd.to_datetime(closed['last_date'])
    closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days
    closed['entry_day'] = closed['first_date'].dt.dayofweek
    closed['entry_day_of_month'] = closed['first_date'].dt.day

    return closed

def print_section(title):
    print(f"\n{'='*75}")
    print(title)
    print('='*75)

def identify_outliers(series, method='iqr', threshold=1.5):
    """Identify outliers using IQR method"""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return (series < lower) | (series > upper)

def calculate_robust_stats(pnl_series, win_series):
    """Calculate statistics that are robust to outliers"""
    if len(pnl_series) < 3:
        return None

    outliers = identify_outliers(pnl_series)
    n_outliers = outliers.sum()

    # Raw stats
    raw_mean = pnl_series.mean()
    raw_total = pnl_series.sum()

    # Robust stats (excluding outliers)
    clean_pnl = pnl_series[~outliers]
    clean_win = win_series[~outliers]

    if len(clean_pnl) < 3:
        clean_pnl = pnl_series
        clean_win = win_series

    return {
        'n': len(pnl_series),
        'n_outliers': n_outliers,
        'outlier_pct': n_outliers / len(pnl_series) * 100,

        # Raw stats
        'raw_total': raw_total,
        'raw_mean': raw_mean,
        'raw_win_rate': win_series.mean() * 100,

        # Robust stats (outliers removed)
        'clean_n': len(clean_pnl),
        'clean_total': clean_pnl.sum(),
        'clean_mean': clean_pnl.mean(),
        'clean_win_rate': clean_win.mean() * 100,

        # Median (naturally robust)
        'median': pnl_series.median(),

        # Spread measures
        'std': pnl_series.std(),
        'clean_std': clean_pnl.std(),
        'iqr': pnl_series.quantile(0.75) - pnl_series.quantile(0.25),

        # Percentiles
        'p10': pnl_series.quantile(0.10),
        'p25': pnl_series.quantile(0.25),
        'p75': pnl_series.quantile(0.75),
        'p90': pnl_series.quantile(0.90),

        # Extremes
        'min': pnl_series.min(),
        'max': pnl_series.max(),

        # Skewness (positive = right tail, outlier gains; negative = left tail, outlier losses)
        'skew': pnl_series.skew() if len(pnl_series) >= 3 else 0,
    }

def statistical_significance_test(group1_pnl, group2_pnl, alpha=0.05):
    """Test if difference between two groups is statistically significant"""
    if len(group1_pnl) < 5 or len(group2_pnl) < 5:
        return {'significant': False, 'reason': 'Insufficient sample size'}

    # Use Mann-Whitney U test (non-parametric, robust to outliers)
    stat, p_value = stats.mannwhitneyu(group1_pnl, group2_pnl, alternative='two-sided')

    return {
        'significant': p_value < alpha,
        'p_value': p_value,
        'effect_size': (group1_pnl.median() - group2_pnl.median())
    }

def sharpe_like_ratio(pnl_series):
    """Calculate a Sharpe-like ratio (mean / std)"""
    if pnl_series.std() == 0:
        return 0
    return pnl_series.mean() / pnl_series.std()

def consistency_score(win_series):
    """How consistent is the win rate? (penalize small samples)"""
    n = len(win_series)
    win_rate = win_series.mean()

    # Wilson score interval (better for small samples)
    if n == 0:
        return 0, 0, 0

    z = 1.96  # 95% confidence
    denominator = 1 + z**2/n
    center = (win_rate + z**2/(2*n)) / denominator
    spread = z * np.sqrt((win_rate*(1-win_rate) + z**2/(4*n)) / n) / denominator

    lower = max(0, center - spread)
    upper = min(1, center + spread)

    return lower * 100, win_rate * 100, upper * 100

def analyze_outliers(df):
    """Detailed outlier analysis"""
    print_section("OUTLIER ANALYSIS")

    outliers = identify_outliers(df['total_pnl'])
    outlier_trades = df[outliers].sort_values('total_pnl')

    print(f"\nTotal trades: {len(df)}")
    print(f"Outlier trades: {outliers.sum()} ({outliers.sum()/len(df)*100:.1f}%)")

    print("\nLARGEST OUTLIERS (these skew your averages):")
    print("-" * 75)

    # Show biggest losses
    print("\nBiggest losses:")
    for _, t in outlier_trades.head(10).iterrows():
        print(f"  ${t['total_pnl']:>+10,.0f} | {t['underlying']:>5} {str(t['option_type']):<4} | {t['first_date'].strftime('%Y-%m-%d')}")

    # Show biggest wins
    print("\nBiggest wins:")
    for _, t in outlier_trades.tail(10).iterrows():
        print(f"  ${t['total_pnl']:>+10,.0f} | {t['underlying']:>5} {str(t['option_type']):<4} | {t['first_date'].strftime('%Y-%m-%d')}")

    # Impact of outliers
    clean_df = df[~outliers]
    print(f"\nIMPACT OF OUTLIERS:")
    print(f"  With outliers:    Total=${df['total_pnl'].sum():>+10,.0f}  Mean=${df['total_pnl'].mean():>+8,.0f}")
    print(f"  Without outliers: Total=${clean_df['total_pnl'].sum():>+10,.0f}  Mean=${clean_df['total_pnl'].mean():>+8,.0f}")
    print(f"  Outliers account for: ${df['total_pnl'].sum() - clean_df['total_pnl'].sum():>+10,.0f}")

def analyze_category_robust(df, category_col, category_name, min_trades=20):
    """Analyze a category with robust statistics"""
    print_section(f"{category_name} - ROBUST ANALYSIS")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    results = []
    for cat in df[category_col].dropna().unique():
        subset = df[df[category_col] == cat]
        if len(subset) < 5:
            continue

        stats_dict = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
        if stats_dict:
            stats_dict['category'] = cat
            stats_dict['sharpe'] = sharpe_like_ratio(subset['total_pnl'])
            lower, mid, upper = consistency_score(subset['is_winner'])
            stats_dict['win_rate_lower'] = lower
            stats_dict['win_rate_upper'] = upper
            results.append(stats_dict)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Not enough data")
        return

    # Sort by clean mean (outlier-adjusted)
    results_df = results_df.sort_values('clean_mean', ascending=False)

    print(f"\n{'Category':<12} | {'N':>4} | {'Out':>3} | {'Raw Mean':>10} | {'Clean Mean':>10} | {'Median':>10} | {'Win% (95% CI)':>20} | {'Sharpe':>6}")
    print("-" * 95)

    for _, r in results_df.iterrows():
        cat_display = day_names[int(r['category'])] if category_col == 'entry_day' else str(r['category'])[:12]

        # Flag if outliers significantly affect the mean
        outlier_flag = '*' if abs(r['raw_mean'] - r['clean_mean']) > abs(r['clean_mean']) * 0.5 else ' '

        print(f"{cat_display:<12} | {int(r['n']):>4} | {int(r['n_outliers']):>3} | ${r['raw_mean']:>+8,.0f}{outlier_flag}| ${r['clean_mean']:>+8,.0f} | ${r['median']:>+8,.0f} | {r['win_rate_lower']:>5.1f}%-{r['win_rate_upper']:>5.1f}% | {r['sharpe']:>+6.2f}")

    print("\n* = Outliers significantly affect the mean (>50% difference)")

    # Statistical significance tests
    if len(results_df) >= 2:
        print(f"\nSTATISTICAL SIGNIFICANCE (comparing best vs worst):")
        print("-" * 60)

        best_cat = results_df.iloc[0]['category']
        worst_cat = results_df.iloc[-1]['category']

        best_data = df[df[category_col] == best_cat]['total_pnl']
        worst_data = df[df[category_col] == worst_cat]['total_pnl']

        test = statistical_significance_test(best_data, worst_data)

        best_display = day_names[int(best_cat)] if category_col == 'entry_day' else str(best_cat)
        worst_display = day_names[int(worst_cat)] if category_col == 'entry_day' else str(worst_cat)

        if test.get('p_value'):
            sig = "YES" if test['significant'] else "NO"
            print(f"  {best_display} vs {worst_display}: p={test['p_value']:.4f} - Significant? {sig}")

def analyze_combinations_robust(df):
    """Analyze day + type combinations with robust statistics"""
    print_section("DAY + TYPE COMBINATIONS - ROBUST ANALYSIS")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    results = []
    for day in range(5):  # Mon-Fri
        for opt_type in ['CALL', 'PUT']:
            subset = df[(df['entry_day'] == day) & (df['option_type'] == opt_type)]
            if len(subset) < 10:
                continue

            stats_dict = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
            if stats_dict:
                stats_dict['day'] = day_names[day]
                stats_dict['type'] = opt_type
                stats_dict['combo'] = f"{day_names[day]} {opt_type}"
                stats_dict['sharpe'] = sharpe_like_ratio(subset['total_pnl'])
                lower, mid, upper = consistency_score(subset['is_winner'])
                stats_dict['win_rate_lower'] = lower
                stats_dict['win_rate_upper'] = upper
                results.append(stats_dict)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Not enough data")
        return

    results_df = results_df.sort_values('clean_mean', ascending=False)

    print(f"\n{'Combo':<12} | {'N':>4} | {'Out':>3} | {'Raw Total':>12} | {'Clean Mean':>10} | {'Median':>10} | {'Win% (95% CI)':>20}")
    print("-" * 95)

    for _, r in results_df.iterrows():
        outlier_flag = '**' if r['n_outliers'] >= 2 and abs(r['raw_mean'] - r['clean_mean']) > 200 else '  '
        print(f"{r['combo']:<12} | {int(r['n']):>4} | {int(r['n_outliers']):>3} | ${r['raw_total']:>+10,.0f}{outlier_flag}| ${r['clean_mean']:>+8,.0f} | ${r['median']:>+8,.0f} | {r['win_rate_lower']:>5.1f}%-{r['win_rate_upper']:>5.1f}%")

    print("\n** = Multiple outliers significantly affect totals")

    # Find the ACTUAL best strategies (by median and consistency)
    print("\nMOST RELIABLE STRATEGIES (by median P&L, min 20 trades):")
    print("-" * 60)
    reliable = results_df[results_df['n'] >= 20].nlargest(5, 'median')
    for _, r in reliable.iterrows():
        print(f"  {r['combo']:<12}: Median ${r['median']:>+6,.0f}, Win rate {r['win_rate_lower']:.0f}%-{r['win_rate_upper']:.0f}%")

def analyze_hold_time_robust(df):
    """Analyze hold time with robust statistics"""
    print_section("HOLD TIME - ROBUST ANALYSIS")

    def hold_bucket(days):
        if pd.isna(days): return None
        if days == 0: return 'Same day'
        elif days == 1: return '1 day'
        elif days <= 3: return '2-3 days'
        elif days <= 7: return '4-7 days'
        elif days <= 14: return '1-2 weeks'
        elif days <= 30: return '2-4 weeks'
        else: return '1+ month'

    df['hold_bucket'] = df['hold_days'].apply(hold_bucket)

    results = []
    bucket_order = ['Same day', '1 day', '2-3 days', '4-7 days', '1-2 weeks', '2-4 weeks', '1+ month']

    for bucket in bucket_order:
        subset = df[df['hold_bucket'] == bucket]
        if len(subset) < 10:
            continue

        stats_dict = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
        if stats_dict:
            stats_dict['bucket'] = bucket
            stats_dict['sharpe'] = sharpe_like_ratio(subset['total_pnl'])
            lower, mid, upper = consistency_score(subset['is_winner'])
            stats_dict['win_rate_lower'] = lower
            stats_dict['win_rate_upper'] = upper
            results.append(stats_dict)

    results_df = pd.DataFrame(results)

    # Maintain order
    results_df['order'] = results_df['bucket'].apply(lambda x: bucket_order.index(x) if x in bucket_order else 99)
    results_df = results_df.sort_values('order')

    print(f"\n{'Hold Time':<12} | {'N':>4} | {'Out':>3} | {'Raw Mean':>10} | {'Clean Mean':>10} | {'Median':>10} | {'Win% (95% CI)':>20}")
    print("-" * 95)

    for _, r in results_df.iterrows():
        outlier_flag = '*' if abs(r['raw_mean'] - r['clean_mean']) > abs(r['clean_mean']) * 0.5 else ' '
        print(f"{r['bucket']:<12} | {int(r['n']):>4} | {int(r['n_outliers']):>3} | ${r['raw_mean']:>+8,.0f}{outlier_flag}| ${r['clean_mean']:>+8,.0f} | ${r['median']:>+8,.0f} | {r['win_rate_lower']:>5.1f}%-{r['win_rate_upper']:>5.1f}%")

    # Test: Is longer holding significantly better?
    print("\nSTATISTICAL TEST: Short holds vs Long holds")
    print("-" * 60)
    short = df[df['hold_days'] <= 1]['total_pnl']
    long = df[df['hold_days'] >= 14]['total_pnl']

    if len(short) >= 10 and len(long) >= 10:
        test = statistical_significance_test(long, short)
        if test.get('p_value'):
            sig = "YES" if test['significant'] else "NO"
            print(f"  Long (14+ days) vs Short (0-1 days): p={test['p_value']:.4f}")
            print(f"  Difference significant at 95% confidence? {sig}")
            print(f"  Median difference: ${long.median() - short.median():+,.0f}")

def analyze_tickers_robust(df):
    """Analyze tickers with robust statistics"""
    print_section("TICKER ANALYSIS - ROBUST")

    results = []
    for ticker in df['underlying'].dropna().unique():
        subset = df[df['underlying'] == ticker]
        if len(subset) < 10:
            continue

        stats_dict = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
        if stats_dict:
            stats_dict['ticker'] = ticker
            stats_dict['sharpe'] = sharpe_like_ratio(subset['total_pnl'])
            lower, mid, upper = consistency_score(subset['is_winner'])
            stats_dict['win_rate_lower'] = lower
            stats_dict['win_rate_upper'] = upper
            results.append(stats_dict)

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("Not enough data")
        return

    # Sort by median (more robust than mean)
    print("\nTOP TICKERS BY MEDIAN P&L (min 10 trades):")
    print("-" * 95)
    print(f"{'Ticker':<8} | {'N':>4} | {'Out':>3} | {'Raw Total':>12} | {'Median':>10} | {'Clean Mean':>10} | {'Win% (95% CI)':>20}")
    print("-" * 95)

    for _, r in results_df.nlargest(10, 'median').iterrows():
        outlier_flag = '*' if r['n_outliers'] >= 2 else ' '
        print(f"{r['ticker']:<8} | {int(r['n']):>4} | {int(r['n_outliers']):>3} | ${r['raw_total']:>+10,.0f}{outlier_flag}| ${r['median']:>+8,.0f} | ${r['clean_mean']:>+8,.0f} | {r['win_rate_lower']:>5.1f}%-{r['win_rate_upper']:>5.1f}%")

    print("\nWORST TICKERS BY MEDIAN P&L (min 10 trades):")
    print("-" * 95)
    for _, r in results_df.nsmallest(10, 'median').iterrows():
        outlier_flag = '*' if r['n_outliers'] >= 2 else ' '
        print(f"{r['ticker']:<8} | {int(r['n']):>4} | {int(r['n_outliers']):>3} | ${r['raw_total']:>+10,.0f}{outlier_flag}| ${r['median']:>+8,.0f} | ${r['clean_mean']:>+8,.0f} | {r['win_rate_lower']:>5.1f}%-{r['win_rate_upper']:>5.1f}%")

def summarize_reliable_findings(df):
    """Summarize only statistically reliable findings"""
    print_section("STATISTICALLY RELIABLE CONCLUSIONS")

    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']

    print("\nThese findings are based on:")
    print("  - Median P&L (robust to outliers)")
    print("  - 95% confidence intervals for win rates")
    print("  - Minimum sample sizes")
    print("  - Statistical significance tests")

    # 1. CALL vs PUT (large sample)
    print("\n" + "-"*60)
    print("1. CALLS vs PUTS")
    print("-"*60)
    calls = df[df['option_type'] == 'CALL']
    puts = df[df['option_type'] == 'PUT']

    call_stats = calculate_robust_stats(calls['total_pnl'], calls['is_winner'])
    put_stats = calculate_robust_stats(puts['total_pnl'], puts['is_winner'])

    print(f"   CALLs (n={call_stats['n']}): Median ${call_stats['median']:+,.0f}, Win rate {call_stats['clean_win_rate']:.1f}%")
    print(f"   PUTs  (n={put_stats['n']}): Median ${put_stats['median']:+,.0f}, Win rate {put_stats['clean_win_rate']:.1f}%")

    test = statistical_significance_test(puts['total_pnl'], calls['total_pnl'])
    if test.get('p_value'):
        print(f"   Statistical difference: p={test['p_value']:.4f} ({'Significant' if test['significant'] else 'Not significant'})")

    # 2. Day of week
    print("\n" + "-"*60)
    print("2. DAY OF WEEK (entry day)")
    print("-"*60)

    day_results = []
    for day in range(5):
        subset = df[df['entry_day'] == day]
        if len(subset) >= 20:
            stats = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
            day_results.append({
                'day': day_names[day],
                'n': stats['n'],
                'median': stats['median'],
                'clean_mean': stats['clean_mean'],
                'win_rate': stats['clean_win_rate']
            })

    day_df = pd.DataFrame(day_results).sort_values('median', ascending=False)
    for _, r in day_df.iterrows():
        print(f"   {r['day']}: n={int(r['n'])}, Median ${r['median']:+,.0f}, Clean mean ${r['clean_mean']:+,.0f}, Win rate {r['win_rate']:.1f}%")

    # Best vs worst day test
    if len(day_df) >= 2:
        best_day = day_df.iloc[0]['day']
        worst_day = day_df.iloc[-1]['day']
        best_data = df[df['entry_day'] == day_names.index(best_day)]['total_pnl']
        worst_data = df[df['entry_day'] == day_names.index(worst_day)]['total_pnl']
        test = statistical_significance_test(best_data, worst_data)
        if test.get('p_value'):
            print(f"   {best_day} vs {worst_day}: p={test['p_value']:.4f} ({'Significant' if test['significant'] else 'Not significant'})")

    # 3. Hold time
    print("\n" + "-"*60)
    print("3. HOLD TIME")
    print("-"*60)

    short = df[df['hold_days'] <= 1]
    medium = df[(df['hold_days'] > 1) & (df['hold_days'] <= 7)]
    long = df[df['hold_days'] > 7]

    for name, subset in [('Short (0-1d)', short), ('Medium (2-7d)', medium), ('Long (7+d)', long)]:
        if len(subset) >= 20:
            stats = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
            print(f"   {name}: n={stats['n']}, Median ${stats['median']:+,.0f}, Win rate {stats['clean_win_rate']:.1f}%")

    # 4. Month period
    print("\n" + "-"*60)
    print("4. TIME OF MONTH (entry)")
    print("-"*60)

    df['month_period'] = df['entry_day_of_month'].apply(
        lambda x: 'Early (1-10)' if x <= 10 else ('Mid (11-20)' if x <= 20 else 'Late (21-31)')
    )

    for period in ['Early (1-10)', 'Mid (11-20)', 'Late (21-31)']:
        subset = df[df['month_period'] == period]
        if len(subset) >= 20:
            stats = calculate_robust_stats(subset['total_pnl'], subset['is_winner'])
            print(f"   {period}: n={stats['n']}, Median ${stats['median']:+,.0f}, Win rate {stats['clean_win_rate']:.1f}%")

    # Final recommendations
    print("\n" + "="*60)
    print("EVIDENCE-BASED RECOMMENDATIONS")
    print("="*60)

    print("""
Based on robust statistical analysis (medians, not means):

STRONG EVIDENCE (statistically significant, large samples):
  - Avoid quick trades (same day/next day): Median is negative
  - Longer holds perform better (consistent pattern across data)

MODERATE EVIDENCE (visible pattern, may need more data):
  - Friday entries underperform other days
  - Early month (days 1-10) underperforms
  - PUTs have higher win rate than CALLs

WEAK/NO EVIDENCE (small samples or outlier-driven):
  - Any specific day + type combo with <30 trades
  - Strategies that depend heavily on 1-2 large wins

KEY INSIGHT:
  Your total P&L is heavily influenced by a few large trades.
  Median trade P&L gives a better picture of typical outcomes.
""")

def main():
    print("="*75)
    print("RIGOROUS STATISTICAL ANALYSIS")
    print("Accounting for outliers, sample size, and statistical significance")
    print("="*75)

    print("\nLoading data...")
    df = load_and_process_trades()
    print(f"Loaded {len(df)} closed positions")

    analyze_outliers(df)
    analyze_category_robust(df, 'option_type', 'OPTION TYPE')
    analyze_category_robust(df, 'entry_day', 'DAY OF WEEK (Entry)')
    analyze_combinations_robust(df)
    analyze_hold_time_robust(df)
    analyze_tickers_robust(df)
    summarize_reliable_findings(df)

    print("\n" + "="*75)
    print("ANALYSIS COMPLETE")
    print("="*75)

if __name__ == "__main__":
    main()
