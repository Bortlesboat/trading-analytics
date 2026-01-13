"""
Deep Quantitative Trading Analysis
==================================
Professional-grade statistical measures for trading performance
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, jarque_bera, shapiro
import re
import warnings
warnings.filterwarnings('ignore')

# Import paths from config.py (local) or use defaults
try:
    from config import TRADE_FILES
except ImportError:
    # Update config.py with your actual paths (see config_example.py)
    TRADE_FILES = ["./data/trades.csv"]

def load_data():
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
        option_type = 'CALL' if 'CALL' in action else ('PUT' if 'PUT' in action else None)
        underlying = row['Symbol']
        match = re.search(r'\(([A-Z]+)\)', action)
        if match: underlying = match.group(1)
        return pd.Series({'option_type': option_type, 'underlying': underlying})

    parsed = trades_df.apply(parse_trade, axis=1)
    trades_df = pd.concat([trades_df, parsed], axis=1)
    trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
    trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
    trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')

    symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',
        'Run Date': ['min', 'max'],
        'Action': 'count'
    }).reset_index()
    symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type', 'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions']

    closed = symbol_pnl[(abs(symbol_pnl['net_quantity']) <= 1) & (symbol_pnl['num_transactions'] >= 2)].copy()
    closed['is_winner'] = closed['total_pnl'] > 0
    closed['first_date'] = pd.to_datetime(closed['first_date'])
    closed['last_date'] = pd.to_datetime(closed['last_date'])
    closed['hold_days'] = (closed['last_date'] - closed['first_date']).dt.days

    return closed.sort_values('last_date').reset_index(drop=True)

def section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)

def basic_performance_metrics(df):
    """Core trading performance metrics"""
    section("1. CORE PERFORMANCE METRICS")

    pnl = df['total_pnl']
    winners = df[df['is_winner']]
    losers = df[~df['is_winner']]

    # Basic counts
    n_trades = len(df)
    n_winners = len(winners)
    n_losers = len(losers)
    win_rate = n_winners / n_trades

    # P&L metrics
    total_pnl = pnl.sum()
    gross_profit = winners['total_pnl'].sum()
    gross_loss = abs(losers['total_pnl'].sum())

    avg_win = winners['total_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = abs(losers['total_pnl'].mean()) if len(losers) > 0 else 0

    # Key ratios
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Expectancy (expected value per trade)
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # Edge calculation
    edge = (win_rate * payoff_ratio) - (1 - win_rate)

    print(f"""
    Trades:           {n_trades:,}
    Winners:          {n_winners:,} ({win_rate*100:.1f}%)
    Losers:           {n_losers:,} ({(1-win_rate)*100:.1f}%)

    Total P&L:        ${total_pnl:+,.0f}
    Gross Profit:     ${gross_profit:+,.0f}
    Gross Loss:       ${-gross_loss:,.0f}

    Avg Winner:       ${avg_win:+,.0f}
    Avg Loser:        ${-avg_loss:,.0f}
    Largest Win:      ${pnl.max():+,.0f}
    Largest Loss:     ${pnl.min():+,.0f}

    --- KEY RATIOS ---
    Profit Factor:    {profit_factor:.2f}  (>1 = profitable system)
    Payoff Ratio:     {payoff_ratio:.2f}  (avg win / avg loss)
    Expectancy:       ${expectancy:+,.2f} per trade
    Edge:             {edge:+.2%}

    Interpretation:
    - Profit Factor {profit_factor:.2f}: {'System is profitable' if profit_factor > 1 else 'System is losing money'}
    - You need {1/(1+payoff_ratio)*100:.0f}% win rate to break even with your payoff ratio
    - Your actual win rate is {win_rate*100:.1f}%
    """)

    return {
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'payoff_ratio': payoff_ratio,
        'expectancy': expectancy,
        'edge': edge
    }

def distribution_analysis(df):
    """Analyze the statistical distribution of returns"""
    section("2. RETURN DISTRIBUTION ANALYSIS")

    pnl = df['total_pnl']

    # Basic distribution stats
    mean = pnl.mean()
    median = pnl.median()
    std = pnl.std()
    skewness = pnl.skew()
    kurtosis = pnl.kurtosis()

    # Percentiles
    p5 = pnl.quantile(0.05)
    p25 = pnl.quantile(0.25)
    p75 = pnl.quantile(0.75)
    p95 = pnl.quantile(0.95)

    # Test for normality
    if len(pnl) >= 20:
        jb_stat, jb_pval = jarque_bera(pnl)
        if len(pnl) <= 5000:
            shap_stat, shap_pval = shapiro(pnl.sample(min(len(pnl), 5000)))
        else:
            shap_stat, shap_pval = None, None
    else:
        jb_stat, jb_pval = None, None
        shap_stat, shap_pval = None, None

    print(f"""
    --- CENTRAL TENDENCY ---
    Mean:             ${mean:+,.0f}
    Median:           ${median:+,.0f}
    Mean vs Median:   ${mean - median:+,.0f}  {'(right-skewed, big wins)' if mean > median else '(left-skewed, big losses)'}

    --- DISPERSION ---
    Std Deviation:    ${std:,.0f}
    IQR:              ${p75 - p25:,.0f}  (middle 50% of trades)
    Coef of Var:      {std/abs(mean)*100:.1f}%  (volatility relative to mean)

    --- DISTRIBUTION SHAPE ---
    Skewness:         {skewness:+.2f}  {'(positive = fat right tail, big wins)' if skewness > 0 else '(negative = fat left tail, big losses)'}
    Kurtosis:         {kurtosis:+.2f}  {'(>0 = fat tails, extreme outcomes)' if kurtosis > 0 else '(<0 = thin tails)'}

    --- PERCENTILES ---
    5th percentile:   ${p5:+,.0f}  (worst 5% of trades)
    25th percentile:  ${p25:+,.0f}
    75th percentile:  ${p75:+,.0f}
    95th percentile:  ${p95:+,.0f}  (best 5% of trades)

    --- NORMALITY TESTS ---""")

    if jb_pval is not None:
        print(f"    Jarque-Bera:    p={jb_pval:.4f} {'(NOT normal)' if jb_pval < 0.05 else '(normal)'}")
    if shap_pval is not None:
        print(f"    Shapiro-Wilk:   p={shap_pval:.4f} {'(NOT normal)' if shap_pval < 0.05 else '(normal)'}")

    print("""
    Interpretation:
    - If returns were normally distributed, 95% would fall within 2 std devs
    - Your actual 5th-95th range suggests the distribution is NOT normal
    - Trading returns typically have 'fat tails' (more extreme outcomes than normal)
    """)

    return {'skewness': skewness, 'kurtosis': kurtosis}

def risk_metrics(df):
    """Risk and drawdown analysis"""
    section("3. RISK METRICS")

    pnl = df['total_pnl']
    cumulative = pnl.cumsum()

    # Running maximum
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    # Drawdown metrics
    max_drawdown = drawdown.min()
    avg_drawdown = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0

    # Time in drawdown
    in_drawdown = (drawdown < 0).sum()
    pct_time_in_drawdown = in_drawdown / len(df) * 100

    # Find longest drawdown period
    drawdown_periods = []
    start_idx = None
    for i, dd in enumerate(drawdown):
        if dd < 0 and start_idx is None:
            start_idx = i
        elif dd >= 0 and start_idx is not None:
            drawdown_periods.append(i - start_idx)
            start_idx = None
    if start_idx is not None:
        drawdown_periods.append(len(drawdown) - start_idx)

    longest_drawdown = max(drawdown_periods) if drawdown_periods else 0

    # Value at Risk (VaR) - how much could you lose in worst X% of trades
    var_95 = pnl.quantile(0.05)  # 5% worst case
    var_99 = pnl.quantile(0.01)  # 1% worst case

    # Conditional VaR (Expected Shortfall) - average loss when beyond VaR
    cvar_95 = pnl[pnl <= var_95].mean()
    cvar_99 = pnl[pnl <= var_99].mean()

    # Risk-adjusted returns
    total_return = pnl.sum()
    sharpe_like = pnl.mean() / pnl.std() if pnl.std() > 0 else 0

    # Sortino (only penalizes downside)
    downside_returns = pnl[pnl < 0]
    downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
    sortino = pnl.mean() / downside_std if downside_std > 0 else 0

    # Calmar ratio (return / max drawdown)
    calmar = total_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')

    # Ulcer Index (measures depth and duration of drawdowns)
    ulcer_index = np.sqrt((drawdown ** 2).mean())

    print(f"""
    --- DRAWDOWN ANALYSIS ---
    Maximum Drawdown:     ${max_drawdown:,.0f}
    Average Drawdown:     ${avg_drawdown:,.0f}
    % Time in Drawdown:   {pct_time_in_drawdown:.1f}%
    Longest Drawdown:     {longest_drawdown} trades
    Current Drawdown:     ${drawdown.iloc[-1]:,.0f}

    --- VALUE AT RISK (VaR) ---
    VaR 95%:              ${var_95:,.0f}  (5% of trades lose more than this)
    VaR 99%:              ${var_99:,.0f}  (1% of trades lose more than this)
    CVaR 95%:             ${cvar_95:,.0f}  (avg loss in worst 5%)
    CVaR 99%:             ${cvar_99:,.0f}  (avg loss in worst 1%)

    --- RISK-ADJUSTED RETURNS ---
    Sharpe-like Ratio:    {sharpe_like:.3f}  (return per unit of risk)
    Sortino Ratio:        {sortino:.3f}  (return per unit of downside risk)
    Calmar Ratio:         {calmar:.3f}  (return / max drawdown)
    Ulcer Index:          {ulcer_index:.0f}  (lower = less painful drawdowns)

    Interpretation:
    - Max drawdown of ${abs(max_drawdown):,.0f} means at worst you were down this much from peak
    - You spend {pct_time_in_drawdown:.0f}% of your trading in drawdown (underwater)
    - CVaR is more useful than VaR - it tells you the AVERAGE bad outcome, not just threshold
    """)

    return {'max_drawdown': max_drawdown, 'sharpe': sharpe_like, 'sortino': sortino}

def kelly_criterion(df):
    """Calculate optimal position sizing using Kelly Criterion"""
    section("4. KELLY CRITERION - OPTIMAL POSITION SIZING")

    winners = df[df['is_winner']]
    losers = df[~df['is_winner']]

    win_rate = len(winners) / len(df)
    avg_win = winners['total_pnl'].mean() if len(winners) > 0 else 0
    avg_loss = abs(losers['total_pnl'].mean()) if len(losers) > 0 else 1

    # Kelly formula: f* = (bp - q) / b
    # where b = odds (avg_win/avg_loss), p = win prob, q = lose prob
    b = avg_win / avg_loss if avg_loss > 0 else 0
    p = win_rate
    q = 1 - win_rate

    kelly_full = (b * p - q) / b if b > 0 else 0
    kelly_half = kelly_full / 2  # Half Kelly is more conservative
    kelly_quarter = kelly_full / 4

    print(f"""
    Kelly Criterion calculates the optimal fraction of capital to risk per trade
    to maximize long-term growth while avoiding ruin.

    Your Parameters:
    - Win Rate (p):       {p*100:.1f}%
    - Loss Rate (q):      {q*100:.1f}%
    - Win/Loss Ratio (b): {b:.2f}

    Optimal Position Sizes:
    - Full Kelly:         {kelly_full*100:+.1f}% of capital per trade
    - Half Kelly:         {kelly_half*100:+.1f}% of capital  (recommended)
    - Quarter Kelly:      {kelly_quarter*100:+.1f}% of capital (conservative)
    """)

    if kelly_full < 0:
        print("""
    WARNING: Negative Kelly means you have NEGATIVE EDGE.
    The mathematically optimal strategy is to NOT trade this system,
    or to bet AGAINST your own picks.
    """)
    elif kelly_full > 0.25:
        print(f"""
    NOTE: Full Kelly of {kelly_full*100:.0f}% is aggressive.
    Most professionals use Half Kelly or less to reduce volatility.
    """)
    else:
        print(f"""
    With ${100000:,} account:
    - Full Kelly position:    ${100000 * kelly_full:,.0f}
    - Half Kelly position:    ${100000 * kelly_half:,.0f}
    - Quarter Kelly position: ${100000 * kelly_quarter:,.0f}
    """)

    return kelly_full

def autocorrelation_analysis(df):
    """Test if wins/losses are independent or clustered"""
    section("5. AUTOCORRELATION - ARE YOUR TRADES INDEPENDENT?")

    # Convert wins to 1, losses to 0
    outcomes = df['is_winner'].astype(int).values
    pnl = df['total_pnl'].values

    # Lag-1 autocorrelation for outcomes
    if len(outcomes) > 10:
        outcome_autocorr = np.corrcoef(outcomes[:-1], outcomes[1:])[0, 1]
        pnl_autocorr = np.corrcoef(pnl[:-1], pnl[1:])[0, 1]
    else:
        outcome_autocorr = 0
        pnl_autocorr = 0

    # Runs test - are wins/losses clustered or random?
    n_runs = 1
    for i in range(1, len(outcomes)):
        if outcomes[i] != outcomes[i-1]:
            n_runs += 1

    n_wins = outcomes.sum()
    n_losses = len(outcomes) - n_wins

    # Expected runs under randomness
    expected_runs = (2 * n_wins * n_losses) / (n_wins + n_losses) + 1
    std_runs = np.sqrt((2 * n_wins * n_losses * (2 * n_wins * n_losses - n_wins - n_losses)) /
                       ((n_wins + n_losses)**2 * (n_wins + n_losses - 1)))

    z_runs = (n_runs - expected_runs) / std_runs if std_runs > 0 else 0
    runs_pval = 2 * (1 - norm.cdf(abs(z_runs)))

    # Win rate after win vs after loss
    win_after_win = []
    win_after_loss = []
    for i in range(1, len(outcomes)):
        if outcomes[i-1] == 1:
            win_after_win.append(outcomes[i])
        else:
            win_after_loss.append(outcomes[i])

    wr_after_win = np.mean(win_after_win) if win_after_win else 0
    wr_after_loss = np.mean(win_after_loss) if win_after_loss else 0

    print(f"""
    --- AUTOCORRELATION ---
    Outcome autocorr:     {outcome_autocorr:+.3f}  (do wins predict wins?)
    P&L autocorrelation:  {pnl_autocorr:+.3f}  (do big trades predict big trades?)

    Interpretation:
    - Close to 0 = trades are independent (random)
    - Positive = wins cluster together, losses cluster together
    - Negative = wins and losses alternate

    --- RUNS TEST ---
    Actual runs:          {n_runs}
    Expected runs:        {expected_runs:.0f}
    Z-score:              {z_runs:+.2f}
    P-value:              {runs_pval:.4f}

    {'Trades appear CLUSTERED (streaky)' if z_runs < -1.96 else 'Trades appear ALTERNATING' if z_runs > 1.96 else 'Trades appear RANDOM (independent)'}

    --- CONDITIONAL WIN RATES ---
    Win rate after a WIN:   {wr_after_win*100:.1f}%  ({len(win_after_win)} samples)
    Win rate after a LOSS:  {wr_after_loss*100:.1f}%  ({len(win_after_loss)} samples)
    Difference:             {(wr_after_win - wr_after_loss)*100:+.1f} percentage points
    """)

    # Statistical test for difference
    if len(win_after_win) >= 20 and len(win_after_loss) >= 20:
        contingency = [[sum(win_after_win), len(win_after_win) - sum(win_after_win)],
                       [sum(win_after_loss), len(win_after_loss) - sum(win_after_loss)]]
        chi2, p_val = stats.chi2_contingency(contingency)[:2]
        print(f"    Chi-square test:    p={p_val:.4f} {'(SIGNIFICANT - past affects future)' if p_val < 0.05 else '(not significant)'}")

    return outcome_autocorr

def monte_carlo_simulation(df, n_simulations=10000):
    """Monte Carlo simulation for future performance estimation"""
    section("6. MONTE CARLO SIMULATION")

    pnl = df['total_pnl'].values
    n_trades = len(pnl)

    # Simulate future performance by resampling
    final_pnls = []
    max_drawdowns = []

    for _ in range(n_simulations):
        # Resample with replacement
        simulated = np.random.choice(pnl, size=n_trades, replace=True)
        cumulative = np.cumsum(simulated)
        final_pnls.append(cumulative[-1])

        # Calculate max drawdown
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        max_drawdowns.append(drawdown.min())

    final_pnls = np.array(final_pnls)
    max_drawdowns = np.array(max_drawdowns)

    print(f"""
    Ran {n_simulations:,} simulations of {n_trades} trades each
    (resampling your actual trade distribution)

    --- FINAL P&L DISTRIBUTION ---
    5th percentile:   ${np.percentile(final_pnls, 5):+,.0f}  (worst case)
    25th percentile:  ${np.percentile(final_pnls, 25):+,.0f}
    50th percentile:  ${np.percentile(final_pnls, 50):+,.0f}  (median outcome)
    75th percentile:  ${np.percentile(final_pnls, 75):+,.0f}
    95th percentile:  ${np.percentile(final_pnls, 95):+,.0f}  (best case)

    Probability of profit:     {(final_pnls > 0).mean()*100:.1f}%
    Probability of >$10K loss: {(final_pnls < -10000).mean()*100:.1f}%
    Probability of >$10K gain: {(final_pnls > 10000).mean()*100:.1f}%

    --- MAX DRAWDOWN DISTRIBUTION ---
    5th percentile:   ${np.percentile(max_drawdowns, 5):,.0f}  (worst drawdown)
    50th percentile:  ${np.percentile(max_drawdowns, 50):,.0f}  (typical drawdown)
    95th percentile:  ${np.percentile(max_drawdowns, 95):,.0f}  (mild drawdown)

    Prob of >$50K drawdown:    {(max_drawdowns < -50000).mean()*100:.1f}%
    Prob of >$100K drawdown:   {(max_drawdowns < -100000).mean()*100:.1f}%
    """)

    return final_pnls

def regime_analysis(df):
    """Analyze performance in different market conditions"""
    section("7. REGIME ANALYSIS")

    df = df.copy()
    df['month'] = df['last_date'].dt.to_period('M')
    df['year'] = df['last_date'].dt.year

    # Monthly stats
    monthly = df.groupby('month').agg({
        'total_pnl': ['sum', 'count', 'mean', 'std'],
        'is_winner': 'mean'
    }).reset_index()
    monthly.columns = ['month', 'pnl', 'trades', 'avg_trade', 'std_trade', 'win_rate']

    # Classify months
    monthly['regime'] = pd.cut(monthly['pnl'],
                               bins=[-float('inf'), -5000, 0, 5000, float('inf')],
                               labels=['Bad', 'Slight Loss', 'Slight Win', 'Good'])

    print("    MONTHLY PERFORMANCE BREAKDOWN:")
    print("    " + "-"*50)
    for regime in ['Good', 'Slight Win', 'Slight Loss', 'Bad']:
        subset = monthly[monthly['regime'] == regime]
        if len(subset) > 0:
            print(f"    {regime:<12}: {len(subset):>2} months, avg ${subset['pnl'].mean():>+8,.0f}/month, {subset['win_rate'].mean()*100:.0f}% win rate")

    # Performance consistency
    profitable_months = (monthly['pnl'] > 0).sum()
    total_months = len(monthly)

    print(f"""
    --- CONSISTENCY ---
    Profitable months:    {profitable_months}/{total_months} ({profitable_months/total_months*100:.0f}%)
    Best month:           ${monthly['pnl'].max():+,.0f}
    Worst month:          ${monthly['pnl'].min():+,.0f}
    Monthly P&L std dev:  ${monthly['pnl'].std():,.0f}

    --- TRADE FREQUENCY ---
    Avg trades/month:     {monthly['trades'].mean():.0f}
    Max trades/month:     {monthly['trades'].max():.0f}
    Min trades/month:     {monthly['trades'].min():.0f}
    """)

    # Correlation between trade frequency and performance
    corr = monthly['trades'].corr(monthly['pnl'])
    print(f"    Correlation (# trades vs P&L): {corr:+.3f}")
    if corr < -0.2:
        print("    >> More trades correlates with WORSE performance")
    elif corr > 0.2:
        print("    >> More trades correlates with BETTER performance")
    else:
        print("    >> No strong relationship between frequency and performance")

def behavioral_analysis(df):
    """Analyze behavioral patterns and biases"""
    section("8. BEHAVIORAL ANALYSIS")

    df = df.copy().sort_values('last_date').reset_index(drop=True)

    # Position sizing after wins/losses
    df['prev_outcome'] = df['is_winner'].shift(1)
    df['prev_pnl'] = df['total_pnl'].shift(1)
    df['position_size'] = df['total_pnl'].abs()  # Proxy for position size

    # Size after win vs loss
    after_win = df[df['prev_outcome'] == True]
    after_loss = df[df['prev_outcome'] == False]

    avg_size_after_win = after_win['position_size'].mean() if len(after_win) > 0 else 0
    avg_size_after_loss = after_loss['position_size'].mean() if len(after_loss) > 0 else 0

    # Performance degradation analysis
    # Split into quintiles by trade number
    df['trade_quintile'] = pd.qcut(range(len(df)), 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    print("    --- POSITION SIZING BEHAVIOR ---")
    print(f"    Avg trade size after WIN:   ${avg_size_after_win:,.0f}")
    print(f"    Avg trade size after LOSS:  ${avg_size_after_loss:,.0f}")

    if avg_size_after_loss > avg_size_after_win * 1.1:
        print("    >> WARNING: You size UP after losses (revenge trading?)")
    elif avg_size_after_win > avg_size_after_loss * 1.1:
        print("    >> You size UP after wins (confidence/overconfidence?)")
    else:
        print("    >> Position sizing is relatively consistent")

    # Performance by quintile
    print("\n    --- PERFORMANCE OVER TIME ---")
    print("    (Is there skill degradation or improvement?)")
    print("    " + "-"*50)

    for q in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
        subset = df[df['trade_quintile'] == q]
        wr = subset['is_winner'].mean() * 100
        avg = subset['total_pnl'].mean()
        print(f"    {q} ({len(subset):>3} trades): {wr:>5.1f}% win rate, ${avg:>+8,.0f} avg")

    # Disposition effect: tendency to sell winners too early, hold losers too long
    print("\n    --- DISPOSITION EFFECT CHECK ---")
    winners = df[df['is_winner']]
    losers = df[~df['is_winner']]

    print(f"    Avg hold time for WINNERS: {winners['hold_days'].mean():.1f} days")
    print(f"    Avg hold time for LOSERS:  {losers['hold_days'].mean():.1f} days")

    if winners['hold_days'].mean() < losers['hold_days'].mean():
        print("    >> Classic disposition effect: cutting winners, holding losers")
    else:
        print("    >> Reverse disposition: holding winners, cutting losers (good!)")

def edge_decay_analysis(df):
    """Analyze if your edge is decaying over time"""
    section("9. EDGE DECAY ANALYSIS")

    df = df.copy().sort_values('last_date').reset_index(drop=True)

    # Rolling metrics
    window = 50  # 50-trade rolling window

    if len(df) < window * 2:
        print("    Not enough trades for edge decay analysis")
        return

    df['rolling_win_rate'] = df['is_winner'].rolling(window).mean()
    df['rolling_avg_pnl'] = df['total_pnl'].rolling(window).mean()
    df['rolling_expectancy'] = df['total_pnl'].rolling(window).apply(
        lambda x: x.mean(), raw=True
    )

    # Compare first half to second half
    mid = len(df) // 2
    first_half = df.iloc[:mid]
    second_half = df.iloc[mid:]

    wr1 = first_half['is_winner'].mean()
    wr2 = second_half['is_winner'].mean()
    avg1 = first_half['total_pnl'].mean()
    avg2 = second_half['total_pnl'].mean()

    print(f"""
    Comparing first {mid} trades to last {len(df)-mid} trades:

                        First Half    Second Half    Change
    Win Rate:           {wr1*100:>6.1f}%       {wr2*100:>6.1f}%       {(wr2-wr1)*100:>+6.1f}%
    Avg Trade P&L:      ${avg1:>+7,.0f}      ${avg2:>+7,.0f}      ${avg2-avg1:>+7,.0f}
    """)

    if wr2 < wr1 - 0.05:
        print("    >> WARNING: Win rate has DECLINED over time")
    elif wr2 > wr1 + 0.05:
        print("    >> GOOD: Win rate has IMPROVED over time")
    else:
        print("    >> Win rate is relatively stable")

    # Linear regression for trend
    x = np.arange(len(df))
    slope, intercept, r, p, se = stats.linregress(x, df['is_winner'].astype(float))

    print(f"\n    Trend Analysis:")
    print(f"    Win rate slope: {slope*1000:+.3f} per 1000 trades")
    print(f"    P-value: {p:.4f} {'(significant trend)' if p < 0.05 else '(no significant trend)'}")

def summary_scorecard(metrics):
    """Generate overall trading scorecard"""
    section("10. OVERALL TRADING SCORECARD")

    scores = {}

    # Score each metric
    if metrics.get('profit_factor', 0) >= 1.5:
        scores['Profitability'] = 'A'
    elif metrics.get('profit_factor', 0) >= 1.0:
        scores['Profitability'] = 'C'
    else:
        scores['Profitability'] = 'F'

    if metrics.get('win_rate', 0) >= 0.55:
        scores['Win Rate'] = 'B'
    elif metrics.get('win_rate', 0) >= 0.45:
        scores['Win Rate'] = 'C'
    else:
        scores['Win Rate'] = 'D'

    if metrics.get('sharpe', 0) >= 0.1:
        scores['Risk-Adjusted'] = 'B'
    elif metrics.get('sharpe', 0) >= 0:
        scores['Risk-Adjusted'] = 'C'
    else:
        scores['Risk-Adjusted'] = 'D'

    if abs(metrics.get('max_drawdown', 0)) < 30000:
        scores['Drawdown'] = 'B'
    elif abs(metrics.get('max_drawdown', 0)) < 60000:
        scores['Drawdown'] = 'C'
    else:
        scores['Drawdown'] = 'F'

    print("""
    METRIC              GRADE    NOTES
    ------------------------------------------------""")
    for metric, grade in scores.items():
        print(f"    {metric:<20}{grade:<9}")

    print("""
    ------------------------------------------------

    Grade Scale:
    A = Excellent, B = Good, C = Average, D = Below Average, F = Poor
    """)

def main():
    print("="*70)
    print(" DEEP QUANTITATIVE TRADING ANALYSIS")
    print("="*70)

    df = load_data()
    print(f"\nLoaded {len(df)} closed trades")
    print(f"Period: {df['first_date'].min().strftime('%Y-%m-%d')} to {df['last_date'].max().strftime('%Y-%m-%d')}")

    # Run all analyses
    perf_metrics = basic_performance_metrics(df)
    dist_metrics = distribution_analysis(df)
    risk_metrics_dict = risk_metrics(df)
    kelly = kelly_criterion(df)
    autocorr = autocorrelation_analysis(df)
    monte_carlo_simulation(df)
    regime_analysis(df)
    behavioral_analysis(df)
    edge_decay_analysis(df)

    # Combine metrics for scorecard
    all_metrics = {**perf_metrics, **risk_metrics_dict}
    summary_scorecard(all_metrics)

    print("\n" + "="*70)
    print(" ANALYSIS COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
