import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from collections import deque
import numpy as np

# --- Configuration ---
DB_FILE = 'robinhood_portfolio.db'

# --- Custom Functions for Calculations ---
@st.cache_data
def get_current_holdings(transactions_df):
    """
    Calculates the current open positions and their cost basis using FIFO logic.
    """
    current_open_positions = {}

    instrument_relevant_actions = transactions_df[
        transactions_df['transaction_category'] == 'Trade'
    ].copy()

    if instrument_relevant_actions.empty:
        return pd.DataFrame(columns=['instrument', 'quantity', 'cost_basis_total'])

    grouped_by_instrument = instrument_relevant_actions.groupby('instrument')

    for instrument_name, group in grouped_by_instrument:
        buy_lots_for_instrument = deque()
        for _, row in group.iterrows():
            trans_code = row['trans_code']
            quantity = row['quantity']
            price = row['price']
            
            if trans_code == 'Buy':
                if pd.notna(quantity) and quantity > 0 and pd.notna(price):
                     buy_lots_for_instrument.append({'quantity': quantity, 'price': price})
            elif trans_code == 'Sell':
                if pd.notna(quantity) and quantity > 0:
                    sold_quantity_remaining = quantity
                    while sold_quantity_remaining > 1e-9 and buy_lots_for_instrument:
                        oldest_lot = buy_lots_for_instrument[0]
                        quantity_to_sell_from_lot = min(sold_quantity_remaining, oldest_lot['quantity'])
                        sold_quantity_remaining -= quantity_to_sell_from_lot
                        oldest_lot['quantity'] -= quantity_to_sell_from_lot
                        if oldest_lot['quantity'] < 1e-9:
                            buy_lots_for_instrument.popleft()

        if buy_lots_for_instrument:
            total_current_quantity = sum(lot['quantity'] for lot in buy_lots_for_instrument)
            total_current_cost_basis = sum(lot['quantity'] * lot['price'] for lot in buy_lots_for_instrument)
            if abs(total_current_quantity) > 1e-9:
                 current_open_positions[instrument_name] = {
                    'quantity': total_current_quantity,
                    'cost_basis_total': total_current_cost_basis
                }

    final_holdings_list = [
        {'instrument': ticker, 'quantity': details['quantity'], 'cost_basis_total': details['cost_basis_total']}
        for ticker, details in current_open_positions.items()
    ]
    current_holdings_df = pd.DataFrame(final_holdings_list)
    return current_holdings_df

# --- Step 1: Connect to the Database and Load Data ---
st.header("Robinhood Portfolio Analysis")
st.markdown("---")

try:
    conn = sqlite3.connect(DB_FILE)
    daily_portfolio_df = pd.read_sql_query("SELECT * FROM daily_portfolio_snapshots", conn)
    closed_trades_df = pd.read_sql_query("SELECT * FROM closed_trades_summary", conn)
    transactions_cleaned_df = pd.read_sql_query("SELECT * FROM transactions_cleaned", conn)
    conn.close()

    # Prepare dataframes
    daily_portfolio_df['Date'] = pd.to_datetime(daily_portfolio_df['Date'])
    daily_portfolio_df.set_index('Date', inplace=True)
    daily_portfolio_df.sort_index(inplace=True)
    
    closed_trades_df['sell_date'] = pd.to_datetime(closed_trades_df['sell_date'])
    transactions_cleaned_df['activity_date'] = pd.to_datetime(transactions_cleaned_df['activity_date'])

    # --- FIX: Calculate current holdings once, making it available to all sections ---
    current_holdings_df = get_current_holdings(transactions_cleaned_df)

    st.success(f"Successfully loaded data from '{DB_FILE}'")

except Exception as e:
    st.error(f"An error occurred during data loading: {e}. Please ensure '{DB_FILE}' exists and the data pipeline has been run.")
    st.stop()

# --- Step 2: V2 Summary Metrics & Key Insights ---
st.subheader("Performance Summary")

# Metric Calculations
overall_twr = (daily_portfolio_df['cumulative_twr_factor'].iloc[-1] - 1) if not daily_portfolio_df.empty else 0
max_drawdown = daily_portfolio_df['drawdown'].min() if not daily_portfolio_df.empty else 0

if not daily_portfolio_df.empty and 'daily_return_adjusted' in daily_portfolio_df.columns and len(daily_portfolio_df) > 1:
    volatility = daily_portfolio_df['daily_return_adjusted'].std() * np.sqrt(252)
    risk_free_rate = 0.02
    daily_risk_free_rate = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = daily_portfolio_df['daily_return_adjusted'] - daily_risk_free_rate
    sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
else:
    volatility = 0
    sharpe_ratio = 0

# Display Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Overall Time-Weighted Return", f"{overall_twr * 100:.2f}%")
col2.metric("Maximum Drawdown", f"{max_drawdown * 100:.2f}%")
col3.metric("Annualized Volatility", f"{volatility * 100:.2f}%")
col4.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

# --- Trading Performance Insights ---
st.subheader("Trading Performance Insights")
if not closed_trades_df.empty:
    winning_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] > 0]
    losing_trades = closed_trades_df[closed_trades_df['realized_profit_loss'] < 0]
    total_trades = len(closed_trades_df)
    win_count = len(winning_trades)
    win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
    avg_win = winning_trades['realized_profit_loss'].mean() if not winning_trades.empty else 0
    avg_loss = losing_trades['realized_profit_loss'].mean() if not losing_trades.empty else 0
    total_gains = winning_trades['realized_profit_loss'].sum()
    total_losses = abs(losing_trades['realized_profit_loss'].sum())
    
    if total_losses > 0:
        profit_factor = total_gains / total_losses
    elif total_gains > 0:
        profit_factor = float('inf')
    else:
        profit_factor = 0

    col1, col2, col3 = st.columns(3)
    col1.metric("Win Rate", f"{win_rate:.2f}%", help="The percentage of closed trades that were profitable.")
    col2.metric("Avg. Gain / Loss", f"${avg_win:,.2f} / ${abs(avg_loss):,.2f}", help="The average dollar amount for winning and losing trades.")
    col3.metric("Profit Factor", f"{profit_factor:.2f}", help="Total gains divided by total losses. A value > 1 indicates profitability.")
else:
    st.info("No closed trades found to calculate trading performance.")

# --- Current Portfolio Holdings ---
st.subheader("Current Portfolio Holdings")
try:
    conn = sqlite3.connect(DB_FILE)
    holdings_summary_df = pd.read_sql_query("SELECT * FROM current_holdings_summary", conn)
    conn.close()

    if not holdings_summary_df.empty:
        total_market_value = holdings_summary_df['market_value'].sum()
        holdings_summary_df['portfolio_allocation_pct'] = (holdings_summary_df['market_value'] / total_market_value) * 100
        holdings_summary_df['unrealized_pl_pct'] = (holdings_summary_df['unrealized_pl'] / holdings_summary_df['cost_basis_total']) * 100
        
        st.dataframe(
            holdings_summary_df[['instrument', 'quantity', 'avg_cost_price', 'current_price', 'market_value', 'unrealized_pl', 'unrealized_pl_pct', 'portfolio_allocation_pct']].sort_values(by='market_value', ascending=False),
            column_config={ "quantity": st.column_config.NumberColumn(format="%.4f"), "avg_cost_price": st.column_config.NumberColumn("Avg Cost", format="$%.2f"), "current_price": st.column_config.NumberColumn("Current Price", format="$%.2f"), "market_value": st.column_config.NumberColumn("Market Value", format="$%,.2f"), "unrealized_pl": st.column_config.NumberColumn("Unrealized P/L", format="$%,.2f"), "unrealized_pl_pct": st.column_config.NumberColumn("Unrealized P/L %", format="%.2f%%"), "portfolio_allocation_pct": st.column_config.NumberColumn("Allocation %", format="%.2f%%"), },
            use_container_width=True
        )
    else:
        st.info("No current holdings found.")
except Exception as e:
    st.warning(f"Could not load current holdings summary. Please ensure the data pipeline has been run. Error: {e}")

# --- Visualizations ---
st.markdown("---")
st.subheader("Portfolio Charts")

tab1, tab2, tab3, tab4 = st.tabs(["Performance vs. Benchmark", "Drawdown", "Portfolio Value", "Sector Allocation"])

with tab1:
    if 'benchmark_cumulative_return' in daily_portfolio_df.columns:
        fig_benchmark = go.Figure()
        fig_benchmark.add_trace(go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['cumulative_twr_factor'], mode='lines', name='My Portfolio', line=dict(color='royalblue', width=2)))
        fig_benchmark.add_trace(go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['benchmark_cumulative_return'], mode='lines', name='S&P 500 (^GSPC)', line=dict(color='grey', width=2, dash='dash')))
        fig_benchmark.update_layout(title='Cumulative Growth: Portfolio vs. Benchmark', xaxis_title='Date', yaxis_title='Cumulative Return Factor (Growth of $1)', legend_title='Legend')
        st.plotly_chart(fig_benchmark, use_container_width=True)
    else:
        st.info("Benchmark comparison data not found.")

with tab2:
    fig_drawdown = go.Figure(data=go.Scatter(x=daily_portfolio_df.index, y=daily_portfolio_df['drawdown'] * 100, fill='tozeroy', mode='lines', line_color='red', name='Drawdown'))
    fig_drawdown.update_layout(title='Portfolio Drawdown Over Time', xaxis_title='Date', yaxis_title='Drawdown (%)')
    st.plotly_chart(fig_drawdown, use_container_width=True)

with tab3:
    fig_portfolio_value = px.line(daily_portfolio_df, y=['Total_Portfolio_Value', 'Cash_Balance', 'Stock_Market_Value'], title='Daily Portfolio Value Components Over Time')
    st.plotly_chart(fig_portfolio_value, use_container_width=True)

with tab4:
    try:
        conn = sqlite3.connect(DB_FILE)
        sector_df = pd.read_sql_query("SELECT * FROM instrument_sectors", conn)
        conn.close()

        if not current_holdings_df.empty and not sector_df.empty:
            holdings_with_sector = pd.merge(current_holdings_df, sector_df, on='instrument', how='left')
            holdings_with_sector['sector'].fillna('N/A', inplace=True)
            sector_allocation = holdings_with_sector.groupby('sector')['cost_basis_total'].sum().reset_index()
            
            fig_pie = px.pie(sector_allocation, names='sector', values='cost_basis_total', title='Sector Allocation by Cost Basis', hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Not enough data to generate sector allocation chart.")
    except Exception as e:
        st.warning(f"Could not generate sector analysis: {e}")

# --- Instrument-Specific Drill-Down Analysis ---
st.markdown("---")
st.subheader("Instrument-Specific Analysis 🔍")

all_instruments = sorted(transactions_cleaned_df['instrument'].dropna().unique().tolist())
selected_instrument = st.selectbox("Select an Instrument to Analyze", all_instruments)

if selected_instrument:
    instrument_transactions = transactions_cleaned_df[transactions_cleaned_df['instrument'] == selected_instrument]
    instrument_closed_trades = closed_trades_df[closed_trades_df['instrument'] == selected_instrument]
    
    st.markdown(f"#### Performance Metrics for **{selected_instrument}**")
    total_realized_pl = instrument_closed_trades['realized_profit_loss'].sum()
    win_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] > 0].shape[0]
    loss_count = instrument_closed_trades[instrument_closed_trades['realized_profit_loss'] < 0].shape[0]
    total_closed_trades = win_count + loss_count
    win_rate_instrument = (win_count / total_closed_trades) * 100 if total_closed_trades > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Realized P/L", f"${total_realized_pl:,.2f}")
    col2.metric("Win Rate", f"{win_rate_instrument:.2f}%")
    col3.metric("Closed Trades", f"{total_closed_trades}")
    
    st.markdown("##### Transaction History")
    st.dataframe(instrument_transactions[['activity_date', 'trans_code', 'quantity', 'price', 'amount']].sort_values(by='activity_date', ascending=False), use_container_width=True)
    
    if not instrument_closed_trades.empty:
        st.markdown("##### Closed Trades Summary")
        st.dataframe(instrument_closed_trades[['sell_date', 'sold_quantity_transaction', 'sell_price', 'realized_profit_loss', 'holding_period_days']].sort_values(by='sell_date', ascending=False), use_container_width=True)
