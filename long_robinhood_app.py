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
        for ticker,
