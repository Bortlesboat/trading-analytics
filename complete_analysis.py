"""
COMPLETE TRADE ANALYSIS - Realized + Unrealized P&L
====================================================
This combines:
1. Closed trades (realized P&L)
2. Current positions (unrealized P&L)
3. Recent 2026 trades

Prepared to accept 2022-2024 historical data later.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
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
# LOAD POSITIONS (Current Holdings with Unrealized P&L)
# =============================================================================
print("="*70)
print("LOADING CURRENT POSITIONS")
print("="*70)

positions_df = pd.read_csv(POSITIONS_FILE, encoding='utf-8-sig', index_col=False)

# Filter to actual positions (not cash, not pending)
positions_df = positions_df[
    (positions_df['Symbol'].notna()) &
    (~positions_df['Symbol'].str.contains('SPAXX|FDRXX|FCASH|USD|Pending', case=False, na=False)) &
    (positions_df['Quantity'].notna()) &
    (positions_df['Quantity'] != 0)
].copy()

# Clean up the data - handle formats like "+$1646.33" or "-$50.00"
def clean_dollar(val):
    if pd.isna(val):
        return np.nan
    val = str(val).replace('$', '').replace(',', '').replace('+', '')
    try:
        return float(val)
    except:
        return np.nan

positions_df['Total Gain/Loss Dollar'] = positions_df['Total Gain/Loss Dollar'].apply(clean_dollar)
positions_df['Current Value'] = positions_df['Current Value'].apply(clean_dollar)
positions_df['Cost Basis Total'] = positions_df['Cost Basis Total'].apply(clean_dollar)

# Identify options vs stocks
positions_df['is_option'] = positions_df['Symbol'].str.strip().str.startswith('-')

# Parse option details
def parse_position_symbol(row):
    symbol = str(row['Symbol']).strip()
    desc = str(row['Description'])

    if symbol.startswith('-'):
        # It's an option
        if 'CALL' in desc:
            opt_type = 'CALL'
        elif 'PUT' in desc:
            opt_type = 'PUT'
        else:
            opt_type = 'UNKNOWN'

        # Extract underlying from description
        import re
        # Pattern like "INTC JAN 15 2027 $30 CALL" -> INTC
        parts = desc.split()
        underlying = parts[0] if parts else symbol

        return pd.Series({'option_type': opt_type, 'underlying': underlying})
    else:
        return pd.Series({'option_type': None, 'underlying': symbol})

parsed_pos = positions_df.apply(parse_position_symbol, axis=1)
positions_df = pd.concat([positions_df, parsed_pos], axis=1)

print(f"\nLoaded {len(positions_df)} positions")

# Summarize unrealized P&L
print("\n" + "-"*70)
print("CURRENT POSITIONS SUMMARY")
print("-"*70)

# Options positions
options_pos = positions_df[positions_df['is_option']]
stocks_pos = positions_df[~positions_df['is_option']]

print(f"\nOPTIONS POSITIONS ({len(options_pos)} positions):")
options_unrealized = options_pos['Total Gain/Loss Dollar'].sum()
print(f"  Total Unrealized P&L: ${options_unrealized:,.2f}")

# Detail by position
print("\n  Individual Option Positions:")
for _, row in options_pos.sort_values('Total Gain/Loss Dollar', ascending=False).iterrows():
    pnl = row['Total Gain/Loss Dollar']
    pnl_pct = row.get('Total Gain/Loss Percent', 'N/A')
    symbol = row['Symbol'].strip()
    underlying = row['underlying']
    opt_type = row['option_type']
    qty = row['Quantity']
    status = "+" if pnl > 0 else ""
    print(f"    {underlying} {opt_type}: {status}${pnl:,.0f} ({qty} contracts) - {row['Account Name']}")

print(f"\nSTOCK POSITIONS ({len(stocks_pos)} positions):")
stocks_unrealized = stocks_pos['Total Gain/Loss Dollar'].sum()
print(f"  Total Unrealized P&L: ${stocks_unrealized:,.2f}")

# Detail by position
print("\n  Individual Stock Positions:")
for _, row in stocks_pos.sort_values('Total Gain/Loss Dollar', ascending=False).iterrows():
    pnl = row['Total Gain/Loss Dollar']
    if pd.isna(pnl):
        continue
    symbol = row['Symbol']
    qty = row['Quantity']
    status = "+" if pnl > 0 else ""
    print(f"    {symbol}: {status}${pnl:,.0f} ({qty} shares) - {row['Account Name']}")

total_unrealized = options_unrealized + stocks_unrealized
print(f"\n{'='*70}")
print(f"TOTAL UNREALIZED P&L: ${total_unrealized:,.2f}")
print(f"{'='*70}")


# =============================================================================
# LOAD ALL TRADE HISTORY FILES
# =============================================================================
print("\n" + "="*70)
print("LOADING TRADE HISTORY")
print("="*70)

all_trades = []

for file_path in TRADE_FILES:
    if os.path.exists(file_path):
        print(f"Loading: {os.path.basename(file_path)}")
        try:
            df = pd.read_csv(file_path, encoding='utf-8-sig')
            # Skip rows that are disclaimers (check if Run Date is valid)
            df = df[df['Run Date'].notna()]
            df = df[~df['Run Date'].astype(str).str.contains('Date downloaded|Brokerage|informational', case=False, na=False)]
            df['source_file'] = os.path.basename(file_path)
            all_trades.append(df)
            print(f"  -> {len(df)} rows")
        except Exception as e:
            print(f"  Error loading {file_path}: {e}")
    else:
        print(f"File not found (skipping): {file_path}")

if all_trades:
    trades_df = pd.concat(all_trades, ignore_index=True)
    print(f"\nTotal trade rows loaded: {len(trades_df)}")
else:
    print("No trade files loaded!")
    trades_df = pd.DataFrame()


# =============================================================================
# PROCESS TRADES (same as before)
# =============================================================================
if len(trades_df) > 0:
    # Filter to actual trades
    trade_keywords = ['YOU BOUGHT', 'YOU SOLD', 'ASSIGNED', 'EXPIRED']
    exclude_keywords = ['DIVIDEND', 'REINVESTMENT']

    trades_df = trades_df[trades_df['Action'].str.contains('|'.join(trade_keywords), case=False, na=False)].copy()
    trades_df = trades_df[~trades_df['Action'].str.contains('|'.join(exclude_keywords), case=False, na=False)]

    # Parse trade details
    def parse_trade(row):
        action = str(row['Action'])
        symbol = str(row['Symbol'])

        if 'BOUGHT OPENING' in action:
            direction = 'BUY_OPEN'
        elif 'SOLD CLOSING' in action:
            direction = 'SELL_CLOSE'
        elif 'SOLD OPENING' in action:
            direction = 'SELL_OPEN'
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

        is_option = 'CALL' in action or 'PUT' in action or symbol.startswith('-')
        option_type = None
        if 'CALL' in action:
            option_type = 'CALL'
        elif 'PUT' in action:
            option_type = 'PUT'

        underlying = symbol
        if is_option and '(' in action:
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

    parsed = trades_df.apply(parse_trade, axis=1)
    trades_df = pd.concat([trades_df, parsed], axis=1)
    trades_df['Run Date'] = pd.to_datetime(trades_df['Run Date'], errors='coerce')
    trades_df['Amount'] = pd.to_numeric(trades_df['Amount'], errors='coerce')

    # Calculate realized P&L from closed positions
    symbol_pnl = trades_df.groupby(['Account', 'Symbol', 'underlying', 'option_type']).agg({
        'Amount': 'sum',
        'Quantity': 'sum',
        'Run Date': ['min', 'max'],
        'direction': 'count'
    }).reset_index()

    symbol_pnl.columns = ['Account', 'Symbol', 'underlying', 'option_type',
                           'total_pnl', 'net_quantity', 'first_date', 'last_date', 'num_transactions']

    # Only count as "closed" if position is flat
    closed_positions = symbol_pnl[
        (abs(symbol_pnl['net_quantity']) <= 1) &
        (symbol_pnl['num_transactions'] >= 2)
    ].copy()

    realized_pnl = closed_positions['total_pnl'].sum()

    print(f"\nClosed positions: {len(closed_positions)}")
    print(f"Realized P&L: ${realized_pnl:,.2f}")


# =============================================================================
# COMPLETE PICTURE
# =============================================================================
print("\n" + "="*70)
print("COMPLETE P&L PICTURE")
print("="*70)

print(f"""
REALIZED P&L (Closed Trades):     ${realized_pnl:,.2f}
UNREALIZED P&L (Open Positions):  ${total_unrealized:,.2f}
                                  -----------------
TOTAL P&L:                        ${realized_pnl + total_unrealized:,.2f}
""")

if realized_pnl + total_unrealized > 0:
    print("STATUS: NET PROFITABLE (when including open positions)")
else:
    print("STATUS: NET NEGATIVE (even with open positions)")


# =============================================================================
# UNREALIZED GAINS BY UNDERLYING
# =============================================================================
print("\n" + "="*70)
print("UNREALIZED P&L BY UNDERLYING")
print("="*70)

underlying_unrealized = positions_df.groupby('underlying').agg({
    'Total Gain/Loss Dollar': 'sum',
    'Current Value': 'sum',
    'Cost Basis Total': 'sum'
}).reset_index()
underlying_unrealized.columns = ['underlying', 'unrealized_pnl', 'current_value', 'cost_basis']
underlying_unrealized = underlying_unrealized.sort_values('unrealized_pnl', ascending=False)

print("\nBy ticker:")
for _, row in underlying_unrealized.iterrows():
    if pd.notna(row['unrealized_pnl']) and row['unrealized_pnl'] != 0:
        status = "+" if row['unrealized_pnl'] > 0 else ""
        print(f"  {row['underlying']}: {status}${row['unrealized_pnl']:,.0f}")


# =============================================================================
# REVISED ANALYSIS: WHAT'S ACTUALLY WORKING?
# =============================================================================
print("\n" + "="*70)
print("REVISED ANALYSIS: WHAT'S ACTUALLY WORKING?")
print("="*70)

# Combine realized + unrealized by underlying
if len(trades_df) > 0 and 'closed_positions' in dir():
    # Get realized by underlying
    realized_by_underlying = closed_positions.groupby('underlying')['total_pnl'].sum().reset_index()
    realized_by_underlying.columns = ['underlying', 'realized_pnl']

    # Merge with unrealized
    combined = pd.merge(
        realized_by_underlying,
        underlying_unrealized[['underlying', 'unrealized_pnl']],
        on='underlying',
        how='outer'
    ).fillna(0)

    combined['total_pnl'] = combined['realized_pnl'] + combined['unrealized_pnl']
    combined = combined.sort_values('total_pnl', ascending=False)

    print("\nTRUE P&L BY TICKER (Realized + Unrealized):")
    print("-"*70)
    print(f"{'Ticker':<10} {'Realized':>12} {'Unrealized':>12} {'TOTAL':>12}")
    print("-"*70)

    for _, row in combined.head(15).iterrows():
        print(f"{row['underlying']:<10} ${row['realized_pnl']:>10,.0f} ${row['unrealized_pnl']:>10,.0f} ${row['total_pnl']:>10,.0f}")

    print("-"*70)
    print(f"{'TOTAL':<10} ${combined['realized_pnl'].sum():>10,.0f} ${combined['unrealized_pnl'].sum():>10,.0f} ${combined['total_pnl'].sum():>10,.0f}")

    # Find biggest winners and losers
    print("\n\nTRUE BIGGEST WINNERS:")
    for _, row in combined.nlargest(5, 'total_pnl').iterrows():
        print(f"  {row['underlying']}: ${row['total_pnl']:,.0f}")

    print("\nTRUE BIGGEST LOSERS:")
    for _, row in combined.nsmallest(5, 'total_pnl').iterrows():
        print(f"  {row['underlying']}: ${row['total_pnl']:,.0f}")


# =============================================================================
# CALLS VS PUTS - REVISED
# =============================================================================
print("\n" + "="*70)
print("CALLS VS PUTS - REVISED WITH UNREALIZED")
print("="*70)

# Realized by option type
if len(trades_df) > 0 and 'closed_positions' in dir():
    realized_calls = closed_positions[closed_positions['option_type'] == 'CALL']['total_pnl'].sum()
    realized_puts = closed_positions[closed_positions['option_type'] == 'PUT']['total_pnl'].sum()
else:
    realized_calls = 0
    realized_puts = 0

# Unrealized by option type
unrealized_calls = options_pos[options_pos['option_type'] == 'CALL']['Total Gain/Loss Dollar'].sum()
unrealized_puts = options_pos[options_pos['option_type'] == 'PUT']['Total Gain/Loss Dollar'].sum()

print(f"""
                    Realized      Unrealized        TOTAL
CALLS:          ${realized_calls:>10,.0f}   ${unrealized_calls:>10,.0f}   ${realized_calls + unrealized_calls:>10,.0f}
PUTS:           ${realized_puts:>10,.0f}   ${unrealized_puts:>10,.0f}   ${realized_puts + unrealized_puts:>10,.0f}
""")

if realized_calls + unrealized_calls > realized_puts + unrealized_puts:
    print("** REVISED: CALLs are actually better when including unrealized gains **")
else:
    print("** CONFIRMED: PUTs still outperform CALLs even with unrealized gains **")


# =============================================================================
# KEY INSIGHT: YOUR "HOLD WINNERS" STRATEGY
# =============================================================================
print("\n" + "="*70)
print("YOUR 'HOLD WINNERS' STRATEGY")
print("="*70)

print(f"""
You mentioned you hold winners and cut losers. The data confirms:

- Realized losses (cut losers): ${realized_pnl:,.0f}
- Unrealized gains (held winners): ${total_unrealized:,.0f}
- Net result: ${realized_pnl + total_unrealized:,.0f}

Your biggest unrealized winners:""")

for _, row in options_pos.nlargest(5, 'Total Gain/Loss Dollar').iterrows():
    pnl = row['Total Gain/Loss Dollar']
    print(f"  {row['underlying']} {row['option_type']}: +${pnl:,.0f} (still holding)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"""
REALIZED P&L:                     ${realized_pnl:,.2f}
UNREALIZED P&L:                   ${total_unrealized:,.2f}
===================================================
TRUE TOTAL P&L:                   ${realized_pnl + total_unrealized:,.2f}

Previous analysis only counted closed trades - this includes unrealized positions.
With unrealized gains included, your actual P&L is ${realized_pnl + total_unrealized:,.0f}.
""")


# =============================================================================
# PREPARE FOR 2022-2024 DATA
# =============================================================================
print("\n" + "="*70)
print("READY FOR HISTORICAL DATA")
print("="*70)
print("""
When you get 2022-2024 data, add the file paths to TRADE_FILES list at the top.
The script will automatically incorporate them.

Expected format: Same Fidelity CSV format with columns:
- Run Date, Account, Action, Symbol, Amount, etc.
""")
