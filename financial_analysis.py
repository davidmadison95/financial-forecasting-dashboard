"""
Financial Performance & Forecasting Dashboard
Main Analysis Script

Analyzes historical financial data, generates forecasts, and creates visualizations
for business decision-making.

Author: David Madison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# Set styling
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class FinancialAnalyzer:
    """
    Financial Performance Analyzer with forecasting capabilities
    """
    
    def __init__(self, data_path):
        """Load and prepare financial data"""
        self.df = pd.read_csv(data_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df = self.df.sort_values('Date')
        print(f"✅ Loaded {len(self.df)} months of financial data")
        print(f"📅 Date Range: {self.df['Date'].min().strftime('%B %Y')} to {self.df['Date'].max().strftime('%B %Y')}")
        
    def calculate_kpis(self):
        """Calculate key performance indicators"""
        kpis = {
            'Total Revenue': self.df['Revenue'].sum(),
            'Total Expenses': self.df['Expenses'].sum(),
            'Total Net Profit': self.df['Net_Profit'].sum(),
            'Avg Monthly Revenue': self.df['Revenue'].mean(),
            'Avg Monthly Profit': self.df['Net_Profit'].mean(),
            'Avg Profit Margin': self.df['Profit_Margin'].mean(),
            'Revenue Growth (Total)': ((self.df['Revenue'].iloc[-1] / self.df['Revenue'].iloc[0]) - 1) * 100,
            'Best Month Revenue': self.df.loc[self.df['Revenue'].idxmax(), 'Date'].strftime('%B %Y'),
            'Worst Month Revenue': self.df.loc[self.df['Revenue'].idxmin(), 'Date'].strftime('%B %Y'),
        }
        return kpis
    
    def analyze_trends(self):
        """Analyze financial trends and patterns"""
        print("\n" + "="*70)
        print("📊 FINANCIAL PERFORMANCE ANALYSIS")
        print("="*70)
        
        kpis = self.calculate_kpis()
        
        print(f"\n💰 Revenue Metrics:")
        print(f"   Total Revenue: ${kpis['Total Revenue']:,.2f}")
        print(f"   Average Monthly Revenue: ${kpis['Avg Monthly Revenue']:,.2f}")
        print(f"   Revenue Growth: {kpis['Revenue Growth (Total)']:.2f}%")
        print(f"   Best Month: {kpis['Best Month Revenue']} (${self.df['Revenue'].max():,.2f})")
        print(f"   Worst Month: {kpis['Worst Month Revenue']} (${self.df['Revenue'].min():,.2f})")
        
        print(f"\n💸 Expense Metrics:")
        print(f"   Total Expenses: ${kpis['Total Expenses']:,.2f}")
        print(f"   Avg Marketing Cost: ${self.df['Marketing_Cost'].mean():,.2f}")
        print(f"   Avg Operational Cost: ${self.df['Operational_Cost'].mean():,.2f}")
        
        print(f"\n📈 Profitability:")
        print(f"   Total Net Profit: ${kpis['Total Net Profit']:,.2f}")
        print(f"   Average Monthly Profit: ${kpis['Avg Monthly Profit']:,.2f}")
        print(f"   Average Profit Margin: {kpis['Avg Profit Margin']:.2f}%")
        
        # Identify anomalies
        profit_std = self.df['Net_Profit'].std()
        profit_mean = self.df['Net_Profit'].mean()
        anomalies = self.df[np.abs(self.df['Net_Profit'] - profit_mean) > 2 * profit_std]
        
        if len(anomalies) > 0:
            print(f"\n⚠️  Anomalies Detected ({len(anomalies)} months):")
            for _, row in anomalies.iterrows():
                print(f"   {row['Date'].strftime('%B %Y')}: ${row['Net_Profit']:,.2f} profit (Margin: {row['Profit_Margin']:.2f}%)")
        
        return kpis
    
    def forecast_revenue(self, periods=12):
        """Forecast future revenue using Prophet"""
        print(f"\n🔮 Forecasting Revenue for next {periods} months...")
        
        # Prepare data for Prophet
        prophet_df = self.df[['Date', 'Revenue']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Split into train/test (use last 6 months as validation)
        train_df = prophet_df[:-6]
        test_df = prophet_df[-6:]
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(train_df)
        
        # Validate on test set
        test_predictions = model.predict(test_df)
        mae = mean_absolute_error(test_df['y'], test_predictions['yhat'])
        rmse = np.sqrt(mean_squared_error(test_df['y'], test_predictions['yhat']))
        mape = mean_absolute_percentage_error(test_df['y'], test_predictions['yhat']) * 100
        
        print(f"   Model Validation (Last 6 Months):")
        print(f"   MAE: ${mae:,.2f}")
        print(f"   RMSE: ${rmse:,.2f}")
        print(f"   MAPE: {mape:.2f}%")
        
        # Make future forecast
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        
        # Extract forecast for new months only
        forecast_new = forecast.tail(periods)
        
        return model, forecast, forecast_new, {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    
    def forecast_expenses(self, periods=12):
        """Forecast future expenses using Prophet"""
        print(f"\n💸 Forecasting Expenses for next {periods} months...")
        
        # Prepare data
        prophet_df = self.df[['Date', 'Expenses']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(prophet_df)
        
        # Forecast
        future = model.make_future_dataframe(periods=periods, freq='MS')
        forecast = model.predict(future)
        forecast_new = forecast.tail(periods)
        
        return model, forecast, forecast_new
    
    def create_visualizations(self, revenue_forecast, expense_forecast, save_dir='visualizations'):
        """Generate all visualizations"""
        print(f"\n📊 Generating visualizations...")
        
        # 1. Revenue vs Expenses Trend
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df['Date'], self.df['Revenue'], label='Revenue', linewidth=2, color='#2E7D32')
        ax.plot(self.df['Date'], self.df['Expenses'], label='Expenses', linewidth=2, color='#C62828')
        ax.fill_between(self.df['Date'], self.df['Revenue'], self.df['Expenses'], 
                        where=(self.df['Revenue'] >= self.df['Expenses']), 
                        alpha=0.2, color='#2E7D32', label='Profit Zone')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
        ax.set_title('Revenue vs Expenses Over Time', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/revenue_vs_expenses.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Profit Margin Trend
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(self.df['Date'], self.df['Profit_Margin'], linewidth=2, color='#1565C0', marker='o', markersize=4)
        ax.axhline(y=self.df['Profit_Margin'].mean(), color='#FF6F00', linestyle='--', 
                   linewidth=2, label=f'Average: {self.df["Profit_Margin"].mean():.2f}%')
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')
        ax.set_title('Profit Margin Trend Analysis', fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/profit_margin_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Revenue Forecast
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Historical data
        ax.plot(self.df['Date'], self.df['Revenue'], label='Historical Revenue', 
                linewidth=2, color='#2E7D32', marker='o', markersize=3)
        
        # Forecast
        forecast_dates = revenue_forecast[1].tail(12)['ds']
        forecast_values = revenue_forecast[1].tail(12)['yhat']
        forecast_lower = revenue_forecast[1].tail(12)['yhat_lower']
        forecast_upper = revenue_forecast[1].tail(12)['yhat_upper']
        
        ax.plot(forecast_dates, forecast_values, label='Forecast', 
                linewidth=2, color='#1565C0', linestyle='--', marker='s', markersize=4)
        ax.fill_between(forecast_dates, forecast_lower, forecast_upper, 
                        alpha=0.2, color='#1565C0', label='95% Confidence Interval')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Revenue ($)', fontsize=12, fontweight='bold')
        ax.set_title('Revenue Forecast (Next 12 Months)', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/revenue_forecast.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Expense Breakdown
        avg_marketing = self.df['Marketing_Cost'].mean()
        avg_operational = self.df['Operational_Cost'].mean()
        
        fig, ax = plt.subplots(figsize=(8, 8))
        colors = ['#FF6F00', '#1565C0']
        explode = (0.05, 0.05)
        ax.pie([avg_marketing, avg_operational], 
               labels=['Marketing', 'Operational'], 
               autopct='%1.1f%%', 
               startangle=90, 
               colors=colors,
               explode=explode,
               textprops={'fontsize': 12, 'fontweight': 'bold'})
        ax.set_title('Average Monthly Expense Breakdown', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/expense_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Year-over-Year Growth
        yearly_data = self.df.groupby('Year').agg({
            'Revenue': 'sum',
            'Net_Profit': 'sum'
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(yearly_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, yearly_data['Revenue'], width, label='Revenue', color='#2E7D32')
        bars2 = ax.bar(x + width/2, yearly_data['Net_Profit'], width, label='Net Profit', color='#1565C0')
        
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Amount ($)', fontsize=12, fontweight='bold')
        ax.set_title('Annual Revenue & Profit Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(yearly_data['Year'])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${height/1000:.0f}K',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/yearly_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Monthly Performance Heatmap
        pivot_data = self.df.pivot_table(
            values='Profit_Margin', 
            index='Month', 
            columns='Year', 
            aggfunc='mean'
        )
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        pivot_data.index = [month_names[i-1] for i in pivot_data.index]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   center=pivot_data.mean().mean(), cbar_kws={'label': 'Profit Margin (%)'},
                   linewidths=0.5, ax=ax)
        ax.set_title('Profit Margin Heatmap by Month & Year', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Month', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/profit_margin_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ 6 visualizations saved to '{save_dir}/' directory")

def main():
    """Main execution function"""
    print("="*70)
    print("💼 FINANCIAL PERFORMANCE & FORECASTING DASHBOARD")
    print("="*70)
    
    # Initialize analyzer
    analyzer = FinancialAnalyzer('data/financials.csv')
    
    # Analyze trends
    kpis = analyzer.analyze_trends()
    
    # Generate forecasts
    revenue_forecast = analyzer.forecast_revenue(periods=12)
    expense_forecast = analyzer.forecast_expenses(periods=12)
    
    # Calculate forecasted profit
    print(f"\n📊 Forecast Summary (Next 12 Months):")
    forecasted_revenue = revenue_forecast[2]['yhat'].sum()
    forecasted_expenses = expense_forecast[2]['yhat'].sum()
    forecasted_profit = forecasted_revenue - forecasted_expenses
    forecasted_margin = (forecasted_profit / forecasted_revenue) * 100
    
    print(f"   Expected Revenue: ${forecasted_revenue:,.2f}")
    print(f"   Expected Expenses: ${forecasted_expenses:,.2f}")
    print(f"   Expected Net Profit: ${forecasted_profit:,.2f}")
    print(f"   Expected Profit Margin: {forecasted_margin:.2f}%")
    
    # Create visualizations
    analyzer.create_visualizations(revenue_forecast, expense_forecast)
    
    # Generate insights
    print("\n" + "="*70)
    print("🧠 KEY INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    # Calculate trends
    recent_6mo = analyzer.df.tail(6)
    prev_6mo = analyzer.df.iloc[-12:-6]
    
    revenue_change = ((recent_6mo['Revenue'].mean() / prev_6mo['Revenue'].mean()) - 1) * 100
    profit_change = ((recent_6mo['Net_Profit'].mean() / prev_6mo['Net_Profit'].mean()) - 1) * 100
    
    print(f"\n1. Recent Performance Trend:")
    if revenue_change > 0:
        print(f"   📈 Revenue increased by {revenue_change:.1f}% in the last 6 months")
    else:
        print(f"   📉 Revenue decreased by {abs(revenue_change):.1f}% in the last 6 months")
    
    print(f"\n2. Profitability Analysis:")
    if analyzer.df['Profit_Margin'].iloc[-1] > analyzer.df['Profit_Margin'].mean():
        print(f"   ✅ Current profit margin ({analyzer.df['Profit_Margin'].iloc[-1]:.2f}%) is above average")
    else:
        print(f"   ⚠️  Current profit margin ({analyzer.df['Profit_Margin'].iloc[-1]:.2f}%) is below average")
    
    print(f"\n3. Forecast Outlook:")
    forecast_growth = ((forecasted_revenue - analyzer.df['Revenue'].iloc[-12:].sum()) / analyzer.df['Revenue'].iloc[-12:].sum()) * 100
    print(f"   Expected {forecast_growth:.1f}% revenue growth over next 12 months")
    
    print(f"\n4. Cost Management:")
    expense_ratio = analyzer.df['Expenses'].iloc[-1] / analyzer.df['Revenue'].iloc[-1]
    if expense_ratio < 0.5:
        print(f"   ✅ Expense ratio is healthy at {expense_ratio*100:.1f}%")
    else:
        print(f"   ⚠️  Expense ratio is elevated at {expense_ratio*100:.1f}% - consider cost optimization")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete! Check 'visualizations/' for detailed charts.")
    print("="*70)

if __name__ == "__main__":
    main()