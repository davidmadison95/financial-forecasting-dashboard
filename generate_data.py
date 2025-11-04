"""
Financial Data Generator
Generates synthetic monthly financial data with realistic patterns, seasonality, and trends.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate date range (Jan 2021 - Oct 2025: 58 months)
start_date = datetime(2021, 1, 1)
months = 58
date_range = pd.date_range(start=start_date, periods=months, freq='MS')

# Base values
base_revenue = 100000
base_marketing = 15000
base_operational = 35000

# Create realistic financial data with trends and seasonality
data = []

for i, date in enumerate(date_range):
    # Add growth trend (5% annual growth = ~0.4% monthly)
    growth_factor = 1 + (i * 0.004)
    
    # Add seasonality (Q4 boost, Q1 dip)
    month = date.month
    if month in [11, 12]:  # Holiday season boost
        seasonal_factor = 1.15
    elif month in [1, 2]:  # Post-holiday dip
        seasonal_factor = 0.92
    elif month in [6, 7, 8]:  # Summer slowdown
        seasonal_factor = 0.96
    else:
        seasonal_factor = 1.0
    
    # Add random variation
    noise = np.random.normal(1, 0.08)
    
    # Calculate revenue
    revenue = base_revenue * growth_factor * seasonal_factor * noise
    
    # Marketing costs (15-20% of revenue with some independence)
    marketing_cost = (base_marketing * growth_factor * 0.9 + 
                      revenue * np.random.uniform(0.12, 0.18))
    
    # Operational costs (more stable, 30-40% of revenue)
    operational_cost = (base_operational * growth_factor * 1.02 + 
                        revenue * np.random.uniform(0.25, 0.32))
    
    # Add occasional cost spikes
    if i in [8, 23, 41]:  # Random months with unusual expenses
        operational_cost *= 1.35
    
    # Total expenses
    expenses = marketing_cost + operational_cost
    
    # Net profit
    net_profit = revenue - expenses
    profit_margin = (net_profit / revenue) * 100
    
    data.append({
        'Date': date,
        'Revenue': round(revenue, 2),
        'Expenses': round(expenses, 2),
        'Marketing_Cost': round(marketing_cost, 2),
        'Operational_Cost': round(operational_cost, 2),
        'Net_Profit': round(net_profit, 2),
        'Profit_Margin': round(profit_margin, 2)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Add derived metrics
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Quarter'] = df['Date'].dt.quarter

# Calculate month-over-month changes
df['Revenue_MoM_Change'] = df['Revenue'].pct_change() * 100
df['Profit_MoM_Change'] = df['Net_Profit'].pct_change() * 100

# Calculate year-over-year changes (12-month lag)
df['Revenue_YoY_Change'] = df['Revenue'].pct_change(periods=12) * 100

# Save to CSV
df.to_csv('data/financials.csv', index=False)

print("✅ Financial data generated successfully!")
print(f"📊 Dataset: {len(df)} months ({df['Date'].min().strftime('%B %Y')} to {df['Date'].max().strftime('%B %Y')})")
print(f"💰 Total Revenue: ${df['Revenue'].sum():,.2f}")
print(f"💸 Total Expenses: ${df['Expenses'].sum():,.2f}")
print(f"📈 Total Profit: ${df['Net_Profit'].sum():,.2f}")
print(f"📉 Average Profit Margin: {df['Profit_Margin'].mean():.2f}%")