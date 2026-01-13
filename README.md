# Trading Analytics System

A Python-based trading analytics system that analyzes brokerage data to identify patterns, behavioral biases, and statistically-validated insights.

## Overview

This project transforms raw trade history into actionable intelligence by answering:
- What's actually working vs what I think is working?
- What behavioral patterns are hurting performance?
- Which statistical patterns are reliable vs noise?

## Features

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `quantitative_analysis.py` | Full quant metrics: Sharpe, Sortino, Kelly, Monte Carlo, drawdowns |
| `statistical_analysis.py` | Outlier-robust analysis with significance tests |
| `complete_stock_options_analysis.py` | Instrument comparison (stocks vs options) |
| `trend_analysis.py` | Pattern finding, streaks, calendar effects |
| `full_history_analysis.py` | Multi-year performance comparison |

### Workflow Tools

| Script | Purpose |
|--------|---------|
| `weekly_review.py` | Automated weekly performance review |
| `trade_checker.py` | Pre-trade checklist scoring system |
| `friday_reminder.py` | Windows notification for weekly reviews |

## Quantitative Metrics

- **Core**: Win rate, profit factor, expectancy, payoff ratio
- **Risk**: Max drawdown, VaR, CVaR, Sharpe, Sortino, Calmar, Ulcer Index
- **Distribution**: Skewness, kurtosis, normality tests
- **Behavioral**: Tilt analysis, revenge trading detection, edge decay
- **Statistical**: Autocorrelation, runs tests, Monte Carlo simulation
- **Sizing**: Kelly Criterion optimal position sizing

## Analytical Framework

### Behavioral Pattern Detection

The system identifies psychological patterns affecting performance:
- **Clustering Effect** - How win rate changes based on previous trade outcome
- **Revenge Trading** - Position size changes after losses
- **Edge Decay** - Performance degradation over time
- **Tilt Detection** - Statistical measurement of emotional trading

### Statistical Validation

All patterns require validation:
- Minimum 50+ trade sample size
- Significance level p < 0.05
- Median-based analysis (reduces outlier impact)
- Chi-square tests for categorical comparisons

### Key Analytical Insights

1. **Outliers dominate** - A small percentage of trades drive most of the P&L
2. **Mean vs Median** - Big losses skew averages; median tells the real story
3. **Small samples lie** - Patterns need sufficient data to be reliable
4. **Behavior > Strategy** - Psychological factors often matter more than stock picks

## Setup

### Requirements
```
pandas
numpy
scipy
```

### Data Format
Expects brokerage CSV exports with standard fields (date, ticker, quantity, price, P&L).

### Usage
```bash
# Full quantitative analysis
python quantitative_analysis.py

# Weekly performance review
python weekly_review.py

# Pre-trade scoring
python trade_checker.py TICKER CALL/PUT
```

## Project Structure

```
trading-analytics/
├── README.md
├── TRADING_RULES.md          # Rule generation framework
├── WEEKLY_WORKFLOW.md        # Review process documentation
├── quantitative_analysis.py  # Deep quant metrics
├── statistical_analysis.py   # Robust statistical tests
├── complete_stock_options_analysis.py
├── trend_analysis.py
├── full_history_analysis.py
├── weekly_review.py
├── trade_checker.py
├── friday_reminder.py
├── review.bat
└── setup_reminder.bat
```

## Tech Stack

- **Python 3.x**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical computations
- **scipy** - Statistical tests and distributions

## License

Personal project. Not financial advice.
