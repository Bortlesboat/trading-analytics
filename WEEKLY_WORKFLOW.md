# WEEKLY TRADING REVIEW WORKFLOW

## One-Time Setup

### 1. Set Up Friday Reminder
Run the setup script to create a Windows notification every Friday at 4:30 PM:
```
setup_reminder.bat
```

### 2. Create Desktop Shortcut (Optional)
1. Right-click on desktop > New > Shortcut
2. Point to `review.bat` in the project folder
3. Name it: "Weekly Trading Review"

---

## Every Friday Workflow

### Step 1: Download Trades (2 minutes)
1. Log into your brokerage
2. Navigate to trade history
3. Set date range: Monday - Friday of this week
4. Download as CSV
5. Save to your configured data directory
6. Rename to: `Accounts_History_YYYY-MM-DD.csv` (use Friday's date)

### Step 2: Run Review (5 minutes)
Either:
- Double-click the desktop shortcut, OR
- Open terminal and run:
```
python weekly_review.py
```

### Step 3: Reflect (5 minutes)
Review the output and ask yourself:
1. Did I follow my trading rules?
2. Did I trade after a loss? (check clustering effect)
3. Did I stick to my preferred instruments?
4. Did I avoid my blacklist?

---

## Quick Reference Commands

```bash
# Weekly review (most recent week)
python weekly_review.py

# Review a specific week
python weekly_review.py 2026-01-13

# Check a trade before entering
python trade_checker.py NVDA PUT
python trade_checker.py AAPL CALL loss

# Full quantitative analysis
python quantitative_analysis.py

# Stocks + Options combined analysis
python complete_stock_options_analysis.py

# Deep trend analysis
python trend_analysis.py

# Statistical analysis (robust to outliers)
python statistical_analysis.py

# Full history analysis
python full_history_analysis.py
```

---

## Key Metrics to Track

The system calculates and tracks:
- Win Rate
- Profit Factor
- Expectancy per Trade
- Average Hold Time
- Clustering Effect (win rate after wins vs losses)
- Position sizing patterns

---

## File Locations

| File | Purpose |
|------|---------|
| `weekly_review.py` | Run every Friday |
| `trade_checker.py` | Check before each trade |
| `quantitative_analysis.py` | Deep quant metrics |
| `statistical_analysis.py` | Outlier-robust stats |
| `complete_stock_options_analysis.py` | Stocks + Options |
| `trend_analysis.py` | Pattern finding |
| `full_history_analysis.py` | Multi-year analysis |
| `review.bat` | One-click weekly review |
| `friday_reminder.py` | Sends notification |
| `TRADING_RULES.md` | Rule generation framework |

---

## Updating Trade Data

When you get new data (new weeks, new years, etc.):

1. Save CSV to your data directory
2. Update `config.py` to add the new file to `TRADE_FILES` list
3. Run analysis to refresh all stats

---

## If You Miss a Week

You can always review past weeks:
```
python weekly_review.py 2026-01-06   # Review that week
python weekly_review.py 2025-12-30   # Review that week
```

The data doesn't go anywhere. Consistency matters more than perfection.

---

## Monthly Deep Dive (First Friday of Month)

Once a month, run the full analysis suite:
```bash
python quantitative_analysis.py
python statistical_analysis.py
python complete_stock_options_analysis.py
```

Check for:
- Edge decay (is your win rate declining?)
- New patterns emerging
- Behavioral drift
