"""
TRADE ANALYSIS - ML on Your Fidelity Options Data
==================================================
This script analyzes your trading history to find patterns.

WHAT IT DOES:
1. Load and clean the Fidelity CSV
2. Match opening/closing transactions to calculate P&L
3. Engineer features (day of week, underlying, option type, etc.)
4. Train a classifier to predict win/loss
5. Show which factors most impact your results
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
    from config import TRADE_FILES
    CSV_PATH = TRADE_FILES[0] if TRADE_FILES else "./data/trades.csv"
except ImportError:
    CSV_PATH = "./data/trades.csv"


# =============================================================================
# STEP 1: Load the data
# =============================================================================
print("Loading trade data...")

df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nDate range: {df['Run Date'].min()} to {df['Run Date'].max()}")


# =============================================================================
# STEP 2: Explore the data
# =============================================================================
print("\n" + "="*60)
print("ACTION TYPES:")
print("="*60)

# Get unique action types
action_counts = df['Action'].value_counts()
for action, count in action_counts.head(20).items():
    print(f"  {count:4d} | {action[:70]}")


# =============================================================================
# STEP 3: Filter to actual trades (exclude dividends, etc.)
# =============================================================================
print("\n" + "="*60)
print("FILTERING TO TRADES...")
print("="*60)

# Keywords that indicate actual trades
trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']

# Filter to only trade rows
trades_df = df[df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)].copy()

# Exclude dividend/reinvestment noise
exclude_keywords = ['DIVIDEND', 'REINVESTMENT']
trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

print(f"Filtered to {len(trades_df)} trade rows (from {len(df)} total)")


# =============================================================================
# STEP 4: Parse trade details
# =============================================================================
print("\n" + "="*60)
print("PARSING TRADE DETAILS...")
print("="*60)

def parse_trade(row):
    """Extract structured info from a trade row."""
    action = row['Action']
    symbol = row['Symbol']

    # Determine trade direction and type
    if 'BOUGHT OPENING' in action:
        direction = 'BUY_OPEN'
    elif 'SOLD CLOSING' in action:
        direction = 'SELL_CLOSE'
    elif 'SOLD OPENING' in action:
        direction = 'SELL_OPEN'  # Writing options
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

    # Determine if it's an option or stock
    is_option = 'CALL' in action or 'PUT' in action or symbol.startswith('-')
    option_type = None
    if 'CALL' in action:
        option_type = 'CALL'
    elif 'PUT' in action:
        option_type = 'PUT'

    # Extract underlying symbol for options
    underlying = symbol
    if is_option and '(' in action:
        # Extract from description like "CALL (AAPL) APPLE INC..."
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


# Apply parsing
parsed = trades_df.apply(parse_trade, axis=1)
trades_df = pd.concat([trades_df, parsed], axis=1)

# Convert date
trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'])

# Extract time features
trades_df['day_of_week'] = trades_df['Run Date'].dt.dayofweek
trades_df['day_name'] = trades_df['Run Date'].dt.day_name()
trades_df['month'] = trades_df['Run Date'].dt.month

print(f"\nTrade types:")
print(trades_df['direction'].value_counts())

print(f"\nOption vs Stock:")
print(trades_df['is_option'].value_counts())

print(f"\nOption types:")
print(trades_df['option_type'].value_counts())

print(f"\nTop underlyings:")
print(trades_df['underlying'].value_counts().head(10))


# =============================================================================
# STEP 5: Calculate P&L per round-trip trade
# =============================================================================
print("\n" + "="*60)
print("CALCULATING ROUND-TRIP P&L...")
print("="*60)

# For options: match OPEN with CLOSE transactions
# Group by account, symbol to find round trips

completed_trades = []

# Focus on options trades with clear open/close
opens = trades_df[trades_df['direction'].isin(['BUY_OPEN', 'SELL_OPEN'])].copy()
closes = trades_df[trades_df['direction'].isin(['SELL_CLOSE', 'BUY_CLOSE', 'EXPIRED', 'ASSIGNED'])].copy()

print(f"Opening transactions: {len(opens)}")
print(f"Closing transactions: {len(closes)}")

# Simple approach: calculate net P&L by symbol and account
# This is imperfect but gives us a starting point

# For each unique (account, symbol) combo, sum the amounts
trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')
trades_df['Quantity'] = pd.to_numeric(trades_df['Quantity'], errors='coerce')
trades_df['Price'] = pd.to_numeric(trades_df['Price'], errors='coerce')

# Group by symbol and account to get net P&L
# Positive Amount = money received (sells)
# Negative Amount = money paid (buys)

symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type']).agg({
    'Amount': 'sum',
    'Quantity': 'sum',
    'Run Date': ['min', 'max'],
    'direction': 'count'
}).reset_index()

symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type',
                       'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions']

# Filter to closed positions (net quantity = 0 or near 0)
# and positions with meaningful activity
closed_positions = symbol_pnl[
    (abs(symbol_pnl['net_quantity']) <= 1) &  # Position is closed or nearly closed
    (symbol_pnl['num_transactions'] >= 2)  # At least open + close
].copy()

print(f"\nFound {len(closed_positions)} closed positions to analyze")

# Classify as win/loss
closed_positions['is_winner'] = closed_positions['total_pnl'] > 0
closed_positions['pnl_dollars'] = closed_positions['total_pnl']

print(f"\nWin/Loss breakdown:")
print(closed_positions['is_winner'].value_counts())

if len(closed_positions) > 0:
    print(f"\nTotal P&L: ${closed_positions['total_pnl'].sum():,.2f}")
    print(f"Average win: ${closed_positions[closed_positions['is_winner']]['total_pnl'].mean():,.2f}")
    print(f"Average loss: ${closed_positions[~closed_positions['is_winner']]['total_pnl'].mean():,.2f}")


# =============================================================================
# STEP 6: Add features for ML
# =============================================================================
print("\n" + "="*60)
print("ENGINEERING FEATURES...")
print("="*60)

# Add time-based features from first trade date
closed_positions['first_date'] = pd.to_datetime(closed_positions['first_date'])
closed_positions['day_of_week'] = closed_positions['first_date'].dt.dayofweek
closed_positions['month'] = closed_positions['first_date'].dt.month

# Hold duration
closed_positions['last_date'] = pd.to_datetime(closed_positions['last_date'])
closed_positions['hold_days'] = (closed_positions['last_date'] - closed_positions['first_date']).dt.days

# Account type
def categorize_account(account):
    if '401k' in account.lower():
        return '401k'
    elif 'roth' in account.lower():
        return 'Roth'
    elif 'traditional' in account.lower():
        return 'Traditional'
    elif 'individual' in account.lower():
        return 'Individual'
    else:
        return 'Other'

closed_positions['account_type'] = closed_positions['Account'].apply(categorize_account)

print("\nFeatures added:")
print(f"  - day_of_week (0=Mon, 6=Sun)")
print(f"  - month")
print(f"  - hold_days")
print(f"  - account_type")
print(f"  - underlying")
print(f"  - option_type")


# =============================================================================
# STEP 7: Basic pattern analysis (before ML)
# =============================================================================
print("\n" + "="*60)
print("PATTERN ANALYSIS")
print("="*60)

if len(closed_positions) >= 10:
    # Win rate by day of week
    print("\nWin rate by DAY OF WEEK:")
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    day_stats = closed_positions.groupby('day_of_week').agg({
        'is_winner': ['sum', 'count', 'mean'],
        'total_pnl': 'sum'
    })
    day_stats.columns = ['wins', 'total', 'win_rate', 'total_pnl']
    for idx, row in day_stats.iterrows():
        if row['total'] > 0:
            print(f"  {day_names[idx]}: {row['win_rate']:.1%} win rate ({int(row['wins'])}/{int(row['total'])}) | ${row['total_pnl']:,.0f}")

    # Win rate by underlying
    print("\nWin rate by UNDERLYING:")
    underlying_stats = closed_positions.groupby('underlying').agg({
        'is_winner': ['sum', 'count', 'mean'],
        'total_pnl': 'sum'
    })
    underlying_stats.columns = ['wins', 'total', 'win_rate', 'total_pnl']
    underlying_stats = underlying_stats.sort_values('total', ascending=False)
    for symbol, row in underlying_stats.head(10).iterrows():
        if row['total'] > 0:
            print(f"  {symbol}: {row['win_rate']:.1%} win rate ({int(row['wins'])}/{int(row['total'])}) | ${row['total_pnl']:,.0f}")

    # Win rate by option type
    print("\nWin rate by OPTION TYPE:")
    option_stats = closed_positions.groupby('option_type').agg({
        'is_winner': ['sum', 'count', 'mean'],
        'total_pnl': 'sum'
    })
    option_stats.columns = ['wins', 'total', 'win_rate', 'total_pnl']
    for opt_type, row in option_stats.iterrows():
        if row['total'] > 0 and opt_type is not None:
            print(f"  {opt_type}: {row['win_rate']:.1%} win rate ({int(row['wins'])}/{int(row['total'])}) | ${row['total_pnl']:,.0f}")

    # Win rate by account type
    print("\nWin rate by ACCOUNT TYPE:")
    account_stats = closed_positions.groupby('account_type').agg({
        'is_winner': ['sum', 'count', 'mean'],
        'total_pnl': 'sum'
    })
    account_stats.columns = ['wins', 'total', 'win_rate', 'total_pnl']
    for acc_type, row in account_stats.iterrows():
        if row['total'] > 0:
            print(f"  {acc_type}: {row['win_rate']:.1%} win rate ({int(row['wins'])}/{int(row['total'])}) | ${row['total_pnl']:,.0f}")

    # Win rate by hold duration
    print("\nWin rate by HOLD DURATION:")
    closed_positions['hold_bucket'] = pd.cut(closed_positions['hold_days'],
                                              bins=[-1, 0, 1, 7, 30, 1000],
                                              labels=['Same day', '1 day', '2-7 days', '8-30 days', '30+ days'])
    hold_stats = closed_positions.groupby('hold_bucket').agg({
        'is_winner': ['sum', 'count', 'mean'],
        'total_pnl': 'sum'
    })
    hold_stats.columns = ['wins', 'total', 'win_rate', 'total_pnl']
    for bucket, row in hold_stats.iterrows():
        if row['total'] > 0:
            print(f"  {bucket}: {row['win_rate']:.1%} win rate ({int(row['wins'])}/{int(row['total'])}) | ${row['total_pnl']:,.0f}")

else:
    print("Not enough closed positions for pattern analysis.")


# =============================================================================
# STEP 8: Train ML Classifier
# =============================================================================
print("\n" + "="*60)
print("TRAINING ML CLASSIFIER")
print("="*60)

if len(closed_positions) >= 20:
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score

    # Prepare features
    feature_cols = ['day_of_week', 'month', 'hold_days']

    # Encode categorical variables
    le_underlying = LabelEncoder()
    le_option = LabelEncoder()
    le_account = LabelEncoder()

    ml_df = closed_positions.copy()
    ml_df['underlying_encoded'] = le_underlying.fit_transform(ml_df['underlying'].fillna('UNKNOWN'))
    ml_df['option_encoded'] = le_option.fit_transform(ml_df['option_type'].fillna('UNKNOWN'))
    ml_df['account_encoded'] = le_account.fit_transform(ml_df['account_type'].fillna('UNKNOWN'))

    feature_cols += ['underlying_encoded', 'option_encoded', 'account_encoded']

    X = ml_df[feature_cols].fillna(0)
    y = ml_df['is_winner'].astype(int)

    print(f"Training on {len(X)} trades with {len(feature_cols)} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X_train, y_train)

    # Evaluate
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nModel Accuracy: {accuracy:.1%}")
    print(f"(Baseline - always guess majority class: {max(y.mean(), 1-y.mean()):.1%})")

    # Feature importance
    print("\nFEATURE IMPORTANCE (what matters most):")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in feature_importance.iterrows():
        bar = '#' * int(row['importance'] * 50)
        print(f"  {row['feature']:20s} {bar} {row['importance']:.1%}")

else:
    print("Need at least 20 closed positions for ML training.")
    print(f"Currently have: {len(closed_positions)}")


# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total rows in CSV: {len(df)}")
print(f"Actual trades: {len(trades_df)}")
print(f"Closed positions analyzed: {len(closed_positions)}")
if len(closed_positions) > 0:
    winners = closed_positions['is_winner'].sum()
    total = len(closed_positions)
    print(f"Overall win rate: {winners/total:.1%} ({winners}/{total})")
    print(f"Total P&L: ${closed_positions['total_pnl'].sum():,.2f}")
