# TRADING RULES FRAMEWORK
## Data-Driven Rule Generation from Trade Analysis

---

## OVERVIEW

This document demonstrates the framework for generating actionable trading rules from quantitative analysis. The methodology transforms raw trade data into statistically-validated behavioral guidelines.

---

## RULE GENERATION METHODOLOGY

### Pre-Trade Checklist Framework

Before each trade, the system evaluates:

1. **Ticker Performance History** - Is this ticker on a blacklist based on historical performance?
2. **Recent Outcome Context** - What was the result of the previous trade?
3. **Hold Time Category** - Does this trade duration match profitable patterns?
4. **Calendar Effects** - Are there time-of-month patterns to consider?

Trades are scored and sized accordingly.

---

## ANALYSIS CATEGORIES

### 1. Instrument Comparison

Compare performance across different instrument types:
- Stocks vs Options
- Calls vs Puts
- Different expiration timeframes

**Metrics evaluated:** Profit factor, average winner/loser ratio, median P&L, hold time

### 2. Behavioral Pattern Detection

Identify psychological patterns affecting performance:
- **Clustering Effect** - How does win rate change based on previous outcome?
- **Revenge Trading** - Does position size increase after losses?
- **Edge Decay** - How has performance changed over time?
- **Tilt Detection** - Statistical measurement of emotional trading

### 3. Ticker-Level Analysis

Categorize tickers into:
- **Whitelist** - Consistently positive expectancy
- **Blacklist** - Consistently negative expectancy
- **Neutral** - Insufficient data or mixed results

### 4. Calendar Effects

Analyze performance by:
- Day of week
- Day of month
- Month of year
- Time since last trade

### 5. Hold Time Analysis

Segment trades by duration:
- Same day
- 1 day
- 2-7 days
- 1-4 weeks
- 1+ months

---

## STATISTICAL VALIDATION

All rules require statistical validation:

| Requirement | Threshold |
|-------------|-----------|
| Minimum sample size | 50+ trades |
| Significance level | p < 0.05 |
| Use median over mean | Reduces outlier impact |
| Chi-square tests | For categorical comparisons |
| Runs tests | For pattern independence |

### Common Statistical Pitfalls

- **Small sample bias** - Patterns with <50 trades are unreliable
- **Survivorship bias** - Long holds may exclude cut losers
- **Outlier domination** - Few large trades can skew means
- **Multiple comparison problem** - Testing many patterns inflates false positives

---

## RULE STRUCTURE

Each rule follows this format:

```
RULE: [Clear directive]
EVIDENCE: [Statistical backing]
ACTION: [Specific behavior change]
EXCEPTION: [When to override]
```

### Example Rule Template

```
RULE: Reduce position size after a losing trade
EVIDENCE: Win rate differs significantly based on previous outcome (p < 0.05)
ACTION: Next trade after a loss should be smaller than previous
EXCEPTION: If >24 hours have passed since the loss
```

---

## CHECKLIST SCORING SYSTEM

Trades are scored on multiple factors:

| Factor | Points |
|--------|--------|
| Favorable instrument type | +1 |
| Ticker on whitelist | +1 |
| Ticker NOT on blacklist | +1 |
| Coming off a win | +1 |
| Favorable calendar period | +1 |
| Appropriate hold time planned | +1 |
| Position size ≤ previous | +1 |

**Scoring thresholds:**
- High score = Full position size
- Medium score = Reduced size
- Low score = Skip trade

---

## TOOLS

```bash
# Weekly performance review
python weekly_review.py

# Pre-trade scoring
python trade_checker.py TICKER CALL/PUT

# Full statistical analysis
python quantitative_analysis.py

# Instrument comparison
python complete_stock_options_analysis.py
```

---

## KEY INSIGHTS

The framework reveals that trading edge often comes from:

1. **Avoiding bad trades** rather than finding good ones
2. **Behavioral discipline** over strategy optimization
3. **Statistical rigor** over intuition
4. **Position sizing** based on recent performance context

---

## IMPLEMENTATION

This rules framework integrates with:
- `trade_checker.py` - Pre-trade scoring
- `weekly_review.py` - Performance tracking
- `quantitative_analysis.py` - Rule validation

The system continuously validates rules against new data to detect edge decay.
