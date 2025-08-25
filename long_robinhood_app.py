import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import numpy as np

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

# --- Custom Functions for Calculations (re-implemented from our V2 pipeline) ---
@st.cache_data
def get_current_holdings(transactions_df):
    """
    Calculates the current open positions and their cost basis using FIFO logic.
    This logic mirrors our Phase 7 realized P/L calculation but stores open lots.
    """
    current_open_positions = {}

    # We only care about Trade category for share quantity changes
    instrument_relevant_actions = transactions_df[
        transactions_df['transaction_category'] == 'Trade'
    ].copy()

    if instrument_relevant_actions.empty:
        return pd.DataFrame(columns=['instrument', 'quantity', 'cost_basis_total'])

    grouped_by_instrument = instrument_relevant_actions.groupby('instrument')

    for instrument_name, group in grouped_by_instrument:
        buy_lots_for_instrument = deque() # Stores {'quantity': float, 'price': float, 'date': datetime}

        for index, row in group.iterrows():
            trans_code = row['trans_code']
            quantity = row['quantity'] # This is the already adjusted quantity
            price = row['price']       # This is the already adjusted price
            activity_date = row['activity_date']

            if trans_code == 'Buy':
                if pd.notna(quantity) and quantity > 0 and pd.notna(price):
                     buy_lots_for_instrument.append({'quantity': quantity, 'price': price, 'date': activity_date})
                # Note: No fallback for missing price/amount here to keep it simple;
                # the pipeline should have already cleaned this up.

            elif trans_code == 'Sell':
                if pd.notna(quantity) and quantity > 0:
                    sold_quantity_remaining = quantity
                    while sold_quantity_remaining > 1e-9 and buy_lots_for_instrument:
                        oldest_lot = buy_lots_for_instrument[0]
                        lot_quantity = oldest_lot['quantity']

                        quantity_to_sell_from_lot = min(sold_quantity_remaining, lot_quantity)

                        sold_quantity_remaining -= quantity_to_sell_from_lot
                        oldest_lot['quantity'] -= quantity_to_sell_from_lot

                        if oldest_lot['quantity'] < 1e-9: # Lot fully consumed or negligible
                            buy_lots_for_instrument.popleft()

        # After processing all transactions for this instrument, sum up remaining lots
        if buy_lots_for_instrument:
            total_current_quantity = sum(lot['quantity'] for lot in buy_lots_for_instrument)
            total_current_cost_basis = sum(lot['quantity'] * lot['price'] for lot in buy_lots_for_instrument)

            if abs(total_current_quantity) > 1e-9:
                 current_open_positions[instrument_name] = {
                    'quantity': total_current_quantity,
                    'cost_basis_total': total_current_cost_basis
                }

    # Convert to DataFrame for display
    final_holdings_list = [
        {'instrument': ticker, 'quantity': details['quantity'], 'cost_basis_total': details['cost_basis_total']}
        for ticker, details in current_open_positions.items()
    ]
    current_holdings_df = pd.DataFrame(final_holdings_list)
    return current_holdings_df


# --- Step 1: Connect to the Database and Load Data ---
st.header("Robinhood Portfolio Analysis")
st.markdown("---")

conn = None
try:
    conn = sqlite3.connect(DB_FILE)

    # Load the daily portfolio snapshots for main charts
    daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])

    # Load the realized P/L summary for performance metrics
    closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)

    # Load the cleaned transactions for cash flow analysis
    transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])

    st.success(f"Successfully loaded data from '{DB_FILE}'")

except sqlite3.Error as e:
    st.error(f"SQLite error: {e}. Please ensure '{DB_FILE}' exists.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during data loading: {e}")
    st.stop()
finally:
    if conn:
        conn.close()

# Ensure dataframes are properly prepared
daily_portfolio_df.set_index('Date', inplace=True)
daily_portfolio_df.sort_index(inplace=True)

closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])

# --- Step 2: V2 Summary Metrics & Key Insights ---
st.subheader("Performance Summary (V2)")

# --- Metric Calculations ---
# Overall TWR
overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) if not daily_portfolio_df.empty else 0

# Max Drawdown
max_drawdown = daily_portfolio_df['drawdown'].min() if not daily_portfolio_df.empty else 0

# Total Realized P/L
total_realized_pl = closed_trades_df['realized_profit_loss'].sum()

# --- NEW: Risk Metric Calculations ---
# Check if we have enough data to calculate risk metrics
if not daily_portfolio_df.empty and 'daily_return_adjusted' in daily_portfolio_df.columns and len(daily_portfolio_df) > 1:
    # 1. Annualized Volatility (Standard Deviation of daily returns)
    # We multiply by sqrt(252) because there are approx. 252 trading days in a year
    volatility = daily_portfolio_df['daily_return_adjusted'].std() * np.sqrt(252)

    # 2. Sharpe Ratio (Risk-Adjusted Return)
    # We need a risk-free rate. Let's assume a constant 2% annual rate for simplicity.
    risk_free_rate = 0.02
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1

    # Calculate excess returns over the risk-free rate
    excess_returns = daily_portfolio_df['daily_return_adjusted'] - daily_risk_free_rate

    # Annualized Sharpe Ratio = mean(excess returns) / std(excess returns) * sqrt(252)
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0

else:
    volatility = 0
    sharpe_ratio = 0


# --- Display Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Time-Weighted Return", f"{overall_twr * 100:.2f}%")
col2.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
col3.metric("Annualized Volatility", f"{volatility * 100:.2f}%")
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

st.markdown("""
<style>
.small-font {
    font-size:0.8rem;
    font-style: italic;
}
</style>
<p class="small-font">Note: Sharpe Ratio calculation assumes a 2% annual risk-free rate.</p>
""", unsafe_allow_html=True)


st.markdown("---")

# --- Step 3: V2 Portfolio Value Visualization ---
st.subheader("Daily Portfolio Value Over Time")
fig_portfolio_value = px.line(daily_portfolio_df,
                               y=['Total_Portfolio_Value', 'Cash_Balance', 'Stock_Market_Value'],
                               title='Daily Portfolio Value Components Over Time (V2)')
st.plotly_chart(fig_portfolio_value, use_container_width=True)

st.markdown("---")

# --- Step 3.5: V2 Portfolio vs. Benchmark Visualization ---
st.subheader("Portfolio Performance vs. S&P 500 Benchmark")

# The 'benchmark_cumulative_return' column should now exist in the dataframe
if 'benchmark_cumulative_return' in daily_portfolio_df.columns:
    # Normalize the starting point to 1 for both portfolio and benchmark
    portfolio_return = daily_portfolio_df['cumulative_twr_factor']
    benchmark_return = daily_portfolio_df['benchmark_cumulative_return']

    fig_benchmark = go.Figure()

    # Add Portfolio TWR trace
    fig_benchmark.add_trace(go.Scatter(
        x=daily_portfolio_df.index,
        y=portfolio_return,
        mode='lines',
        name='My Portfolio',
        line=dict(color='royalblue', width=2)
    ))

    # Add Benchmark trace
    fig_benchmark.add_trace(go.Scatter(
        x=daily_portfolio_df.index,
        y=benchmark_return,
        mode='lines',
        name='S&P 500 (^GSPC)',
        line=dict(color='grey', width=2, dash='dash')
    ))

    fig_benchmark.update_layout(
        title='Cumulative Growth: Portfolio vs. Benchmark',
        xaxis_title='Date',
        yaxis_title='Cumulative Return Factor (Growth of $1)',
        legend_title='Legend'
    )
    st.plotly_chart(fig_benchmark, use_container_width=True)
else:
    st.info("Benchmark comparison data not found. Please re-run the data processing pipeline.")

st.markdown("---")

# --- Step 4: V2 Drawdown Visualization ---
st.subheader("Portfolio Drawdown Over Time")
fig_drawdown = go.Figure(data=go.Scatter(
    x=daily_portfolio_df.index,
    y=daily_portfolio_df['drawdown'] * 100,
    fill='tozeroy',
    mode='lines',
    line_color='red',
    name='Drawdown'
))
fig_drawdown.update_layout(
    title='Portfolio Drawdown Over Time (V2)',
    xaxis_title='Date',
    yaxis_title='Drawdown (%)'
)
st.plotly_chart(fig_drawdown, use_container_width=True)

st.markdown("---")

# --- Step 5: Realized P/L, Holdings, and Trading Performance ---

# (This section remains the same)
st.subheader("Realized P/L by Instrument")
if not closed_trades_df.empty:
    realized_pl_summary = closed_trades_df.groupby('instrument')['realized_profit_loss'].sum().reset_index()
    realized_pl_summary.columns = ['Instrument', 'Total Realized P/L ($)']
    realized_pl_summary.sort_values(by='Total Realized P/L ($)', ascending=False, inplace=True)
    st.dataframe(realized_pl_summary, use_container_width=True)
else:
    st.info("No closed trades found to display realized P/L.")

# (REPLACE your existing "Current Portfolio Holdings" section in robinhood_app.py with this)

st.subheader("Current Portfolio Holdings")

try:
    # Load the new summary table from the database
    conn = sqlite3.connect(DB_FILE)
    holdings_summary_df = pd.read_sql_query("SELECT * FROM current_holdings_summary", conn)
    conn.close()

    if not holdings_summary_df.empty:
        # Calculate portfolio allocation
        total_market_value = holdings_summary_df['market_value'].sum()
        holdings_summary_df['portfolio_allocation_pct'] = (holdings_summary_df['market_value'] / total_market_value) * 100
        
        # Format for display
        holdings_summary_df['unrealized_pl_pct'] = (holdings_summary_df['unrealized_pl'] / holdings_summary_df['cost_basis_total']) * 100

        # Display the detailed table
        st.dataframe(
            holdings_summary_df[[
                'instrument',
                'quantity',
                'avg_cost_price',
                'current_price',
                'market_value',
                'unrealized_pl',
                'unrealized_pl_pct',
                'portfolio_allocation_pct'
            ]].sort_values(by='market_value', ascending=False),
            column_config={
                "quantity": st.column_config.NumberColumn(format="%.4f"),
                "avg_cost_price": st.column_config.NumberColumn("Avg Cost", format="$%.2f"),
                "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Market Value", format="$%,.2f"),
                "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="$%,.2f"),
                "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized P/L %", format="%.2f%%"),
                "portfolio_allocation_pct": st.column_config.NumberColumn("Allocation %", format="%.2f%%"),
            },
            use_container_width=True
        )
    else:
        st.info("No current holdings found.")

except Exception as e:
    st.warning(f"Could not display current holdings: {e}")


# --- START OF NEW SECTION ---
# (Place the new Trading Performance section here)
st.markdown("---")
st.subheader("Trading Performance Insights")

# --- Win/Loss Ratio Calculation ---
if not closed_trades_df.empty:
    winning_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] > 0]
    losing_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] < 0]

    total_trades = len(closed_trades_df)
    win_count = len(winning_trades)

    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # Safely calculate average gain and loss
    avg_win = winning_trades['realized_profit_loss'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['realized_profit_loss'].mean() if not losing_trades.empty else 0

    # Safely calculate Profit Factor
    total_gains = winning_trades['realized_profit_loss'].sum()
    total_losses = abs(losing_trades['realized_profit_loss'].sum())

    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = float('inf') # Only gains, no losses
    else:
        profit_factor = 0 # No gains or losses

    # Display metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{win_rate:.2f}%", help="The percentage of closed trades that were profitable.")
    # Use abs(avg_loss) to display it as a positive number
    col2.metric("Avg. Gain / Loss", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}", help="The average dollar amount for winning and losing trades.")
    col3.metric("Profit Factor", f"{profit_factor:.2f}", help="Total gains divided by total losses. A value > 1 indicates profitability.")

else:
    st.info("No closed trades found to calculate Win/Loss Ratio.")

# --- END OF SCRIPT ---
# (Add this code to robinhood_app.py, for example, after the "Current Holdings" section)


# --- Sector Allocation Analysis ---
st.markdown("---")
st.subheader("Portfolio Sector Allocation")

try:
    # Reload data from the new table
    conn = sqlite3.connect(DB_FILE)
    sector_df = pd.read_sql_query("SELECT * FROM instrument_sectors", conn)
    conn.close()

    # Get current holdings and their market value (we need to estimate this)
    # Note: For a precise market value, you'd need live price data.
    # Here, we'll use the cost basis as a proxy for allocation.
    holdings_with_cost = get_current_holdings(transactions_cleaned_df)

    if not holdings_with_cost.empty and not sector_df.empty:
        # Merge holdings with sector data
        holdings_with_sector = pd.merge(holdings_with_cost, sector_df, on='instrument', how='left')
        holdings_with_sector['sector'].fillna('N/A', inplace=True)

        # Group by sector and sum the cost basis
        sector_allocation = holdings_with_sector.groupby('sector')['cost_basis_total'].sum().reset_index()

        # Create a pie chart
        fig_pie = px.pie(sector_allocation,
                         names='sector',
                         values='cost_basis_total',
                         title='Sector Allocation by Cost Basis',
                         hole=0.3)
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Not enough data to generate sector allocation chart.")

except Exception as e:
    st.warning(f"Could not generate sector analysis: {e}")
# (Add this code to the end of your robinhood_app.py script)

# --- Instrument-Specific Drill-Down Analysis ---
st.markdown("---")
st.subheader("Instrument-Specific Analysis 🔍")

# Get a sorted list of unique instruments from your transaction history
all_instruments = sorted(transactions_cleaned_df['instrument'].dropna().unique().tolist())

# Create a dropdown menu for the user to select an instrument
selected_instrument = st.selectbox("Select an Instrument to Analyze", all_instruments)

if selected_instrument:
    # --- Filter data for the selected instrument ---
    instrument_transactions = transactions_cleaned_df[transactions_cleaned_df['instrument'] == selected_instrument]
    instrument_closed_trades = closed_trades_df[closed_trades_df['instrument'] == selected_instrument]

    # --- Display Key Metrics ---
    st.markdown(f"#### Performance Metrics for **{selected_instrument}**")

    # Calculate metrics
    total_realized_pl = instrument_closed_trades['realized_profit_loss'].sum()
    win_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] > 0].shape[0]
    loss_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] < 0].shape[0]
    total_trades = win_count + loss_count
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

    # Check current holdings for unrealized P/L
    current_holding = current_holdings_df[current_holdings_df['instrument'] == selected_instrument]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Realized P/L", f"${total_realized_pl:,.2f}")
    col2.metric("Win Rate", f"{win_rate:.2f}%")
    col3.metric("Closed Trades", f"{total_trades}")

    # --- Display Current Position (if any) ---
    if not current_holding.empty:
        st.markdown("##### Current Position")
        st.dataframe(current_holding, use_container_width=True)

    # --- Display Transaction History ---
    st.markdown("##### Transaction History")
    # Show relevant columns from the transaction log for this instrument
    st.dataframe(
        instrument_transactions[['activity_date', 'trans_code', 'quantity', 'price', 'amount']].sort_values(
            by='activity_date', ascending=False
        ),
        use_container_width=True
    )

    # --- Display Closed Trades Summary ---
    if not instrument_closed_trades.empty:
        st.markdown("##### Closed Trades Summary")
        st.dataframe(
            instrument_closed_trades[['sell_date', 'sold_quantity_transaction', 'sell_price', 'realized_profit_loss', 'holding_period_days']].sort_values(
                by='sell_date', ascending=False
            ),
            use_container_width=True
        )
