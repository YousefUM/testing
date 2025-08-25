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

# (This section also remains the same)
st.subheader("Current Portfolio Holdings")
current_holdings_df = get_current_holdings(transactions_cleaned_df)
if not current_holdings_df.empty:
    st.dataframe(current_holdings_df.sort_values(by='quantity', ascending=False), use_container_width=True)
else:
    st.info("No current holdings found.")

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

