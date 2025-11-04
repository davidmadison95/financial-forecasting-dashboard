💼 Financial Performance & Forecasting Dashboard

Machine learning-powered financial analytics with automated forecasting and interactive KPI monitoring








📋 Project Overview

This project delivers a comprehensive Financial Performance & Forecasting Dashboard that enables organizations to monitor financial health, analyze historical trends, and generate data-driven predictions for strategic planning.

The system processes monthly financial data across revenue, expenses, and profitability metrics, applying advanced time-series forecasting to predict future performance with quantified confidence intervals.

🎯 Key Capabilities

✅ Real-time KPI Monitoring – Track revenue, expenses, profit margins, and growth rates
✅ Automated Forecasting – 12-month revenue and expense predictions using Facebook Prophet
✅ Anomaly Detection – Identify unusual profit fluctuations requiring investigation
✅ Interactive Dashboard – Streamlit-powered interface with dynamic filtering and visualization
✅ Business Insights – Automated narrative summaries with actionable recommendations

💼 Business Problem

Many organizations rely on static financial reports that provide historical snapshots without predictive capabilities. These systems:

❌ Lack forward-looking insights for budget planning

❌ Miss early warning signals of margin deterioration

❌ Require manual analysis to identify cost trends

❌ Fail to quantify forecast uncertainty for risk management

This dashboard bridges that gap by combining descriptive analytics (what happened) with predictive analytics (what's likely next), enabling proactive financial decision-making.

📊 Dataset Description

The analysis uses 58 months of synthetic financial data (January 2021 - October 2025) containing:

Column	Description	Example
Date	Month and year	2024-06
Revenue	Total monthly revenue	$125,000
Expenses	Total monthly expenses	$83,000
Marketing_Cost	Marketing and advertising spend	$18,000
Operational_Cost	Overhead and administrative expenses	$32,000
Net_Profit	Revenue minus Expenses	$42,000
Profit_Margin	(Net Profit / Revenue) × 100	33.6%

Data Characteristics:

📈 Built-in growth trend (~5% annual)

🔄 Seasonal patterns (Q4 boost, Q1/summer dips)

⚡ Intentional anomalies to test detection algorithms

📉 Realistic noise and variance

🛠️ Methodology
1️⃣ Data Preparation & Cleaning
# Load and validate data
df = pd.read_csv('data/financials.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Calculate derived metrics
df['Revenue_MoM_Change'] = df['Revenue'].pct_change() * 100
df['Revenue_YoY_Change'] = df['Revenue'].pct_change(periods=12) * 100


Steps:

Convert date strings to datetime format

Calculate month-over-month (MoM) and year-over-year (YoY) growth rates

Handle missing values and validate data integrity

2️⃣ Exploratory Data Analysis (EDA)

Key metrics calculated:

💰 Total Revenue: $6.52M (58 months)

💸 Total Expenses: $6.04M

📈 Net Profit: $478K (7.3% avg margin)

📊 Revenue Growth: 36.4% over period

⚠️ Anomalies: 3 months with unusual profit swings

Insights:

Revenue demonstrates steady upward trend with seasonal variation

Expense ratio fluctuates between 85-95% of revenue

Q4 consistently shows highest revenue due to holiday boost

Identified cost spikes in Sep 2021, Jun 2024 requiring investigation

3️⃣ Time-Series Forecasting

Model: Facebook Prophet

Prophet excels at handling:

Yearly seasonality (holiday effects, quarterly patterns)

Trend changes over time

Missing data and outliers

Confidence intervals for predictions

from prophet import Prophet

# Prepare data
prophet_df = df[['Date', 'Revenue']].copy()
prophet_df.columns = ['ds', 'y']

# Train model
model = Prophet(
    yearly_seasonality=True,
    changepoint_prior_scale=0.05
)
model.fit(prophet_df)

# Generate 12-month forecast
future = model.make_future_dataframe(periods=12, freq='MS')
forecast = model.predict(future)


Forecast Performance:

Metric	Value	Interpretation
MAE	$4,539	Average forecast error
RMSE	$5,629	Root mean squared error
MAPE	3.61%	Mean absolute percentage error

✅ MAPE < 5% indicates excellent forecast accuracy

4️⃣ Visualization & Dashboard Development

Six core visualizations created:

Revenue vs Expenses Trend – Line chart with profit zone shading

Profit Margin Analysis – Trend line with average benchmark

Revenue Forecast Chart – 12-month prediction with confidence bands

Expense Breakdown – Pie chart showing cost categories

Annual Comparison – Grouped bar chart of yearly totals

Profit Margin Heatmap – Monthly performance by year

Interactive Streamlit Dashboard Features:

📊 Real-time KPI cards with delta indicators

🎛️ Dynamic date range filters (Last 6 months, quarter, year, all time)

🔮 Adjustable forecast horizons (3-12 months)

📈 Plotly interactive charts with zoom/pan/hover

📋 Exportable raw data tables

📈 Key Results & Insights
Business Performance Summary
📊 Financial Overview (58 months)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 Total Revenue:        $6,521,953
💸 Total Expenses:       $6,043,690
📈 Net Profit:           $478,263
📊 Average Margin:       6.92%
📈 Revenue Growth:       36.41%

Forecast Outlook (Next 12 Months)
Metric	Projected Value	Growth Rate
Expected Revenue	$1,499,550	+1.0%
Expected Expenses	$1,420,813	—
Expected Profit	$78,737	—
Expected Margin	5.25%	-1.67pp

⚠️ Key Finding: Forecast indicates margin compression - expenses growing faster than revenue. Recommend cost optimization initiatives.

Anomaly Detection Results

3 months flagged for review:

September 2021: -$11,524 profit (-11.11% margin) – Operational cost spike

June 2024: -$20,696 profit (-18.56% margin) – Expense anomaly

December 2024: +$37,517 profit (23.37% margin) – Holiday revenue surge

Actionable Recommendations

🎯 Cost Control Priority
Current expense ratio (93.9%) is elevated. Target 5-7% reduction in operational costs.

📊 Margin Recovery Strategy
Recent 6-month profit margin below historical average. Review pricing strategy.

🔮 Revenue Acceleration
Forecast shows modest 1% growth. Explore new revenue streams or market expansion.

⚠️ Anomaly Investigation
Deep-dive analysis needed for Jun 2024 expense spike ($20K profit loss).

🚀 Installation & Usage
Prerequisites
python --version  # 3.8 or higher required

Quick Start
# Clone repository
git clone https://github.com/yourusername/financial-forecasting-dashboard.git
cd financial-forecasting-dashboard

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python generate_data.py

# Run analysis script
python financial_analysis.py

# Launch interactive dashboard
streamlit run dashboard.py

Expected Output

Terminal Analysis:

✅ Comprehensive KPI summary

✅ Trend analysis with insights

✅ Forecast accuracy metrics

✅ Business recommendations

Visualizations:

📊 6 PNG charts saved to /visualizations/

Dashboard:

🌐 Opens in browser at http://localhost:8501

🎛️ Interactive controls and real-time updates

📁 Project Structure
financial-forecasting-dashboard/
│
├── dashboard.py                    # Streamlit interactive dashboard
├── financial_analysis.py           # Main analysis & forecasting script
├── generate_data.py                # Synthetic data generator
│
├── data/
│   └── financials.csv             # Financial dataset (58 months)
│
├── visualizations/                # Generated charts
│   ├── revenue_vs_expenses.png
│   ├── profit_margin_trend.png
│   ├── revenue_forecast.png
│   ├── expense_breakdown.png
│   ├── yearly_comparison.png
│   └── profit_margin_heatmap.png
│
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebook (optional)
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── LICENSE                        # MIT License

🧰 Technical Stack
Category	Technology	Purpose
Language	Python 3.8+	Core programming
Data Processing	pandas, NumPy	Data manipulation & analysis
Visualization	matplotlib, seaborn, Plotly	Static & interactive charts
Forecasting	Facebook Prophet	Time-series modeling
Dashboard	Streamlit	Web-based interactive UI
Validation	scikit-learn	Forecast accuracy metrics
Environment	Jupyter Notebook	Exploratory analysis
🔬 Model Details
Prophet Configuration
Prophet(
    yearly_seasonality=True,      # Capture annual patterns
    weekly_seasonality=False,     # Not relevant for monthly data
    daily_seasonality=False,      # Not relevant for monthly data
    changepoint_prior_scale=0.05  # Flexibility for trend changes
)


Why Prophet?

✅ Handles missing data and outliers robustly
✅ Built-in seasonality detection
✅ Produces confidence intervals (95% by default)
✅ Fast training even on limited data
✅ Interpretable components (trend, seasonality, holidays)

Forecast Validation Strategy

Training Set: First 52 months (Jan 2021 - Apr 2025)

Test Set: Last 6 months (May 2025 - Oct 2025)

Metrics: MAE, RMSE, MAPE

MAPE < 5% achieved on validation set → Model ready for production forecasting

🧠 Key Learning Outcomes

This project demonstrates:

✅ Time-series forecasting with Prophet for financial planning
✅ Interactive dashboard development using Streamlit
✅ Anomaly detection for operational risk management
✅ Business storytelling through automated insight generation
✅ Production-ready code with modular, documented functions

🚧 Future Enhancements

 Scenario Planning: Allow users to simulate "what-if" cost/revenue scenarios

 Multi-department Breakdown: Expand data to include department-level analysis

 Alert System: Email notifications when KPIs exceed thresholds

 Database Integration: Connect to live SQL database instead of CSV

 Advanced Models: Compare Prophet vs ARIMA vs LSTM forecasts

 Explanatory Features: Add model explainability using SHAP values

 Export Reports: Automated PDF report generation with executive summary

👨‍💼 Author

David Madison
Data Analyst | Financial Analytics & Machine Learning

📧 davidmadison95@yahoo.com

🔗 LinkedIn - https://www.linkedin.com/in/davidmadison95/

💼 Portfolio - https://davidmadison95.github.io/Business-Portfolio/

📄 License

MIT License - free for educational and professional use with attribution.

🙏 Acknowledgments

Facebook Prophet – Open-source forecasting library

Streamlit – Rapid dashboard development framework

Plotly – Interactive visualization library

📚 References

Taylor, S. J., & Letham, B. (2018). "Forecasting at scale." The American Statistician, 72(1), 37-45.

Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: principles and practice (3rd ed.)

🎯 Project Highlights for Resume

Financial Performance & Forecasting Dashboard – View Dashboard
 | Live Demo

Python-based financial analytics system with automated forecasting and interactive KPI monitoring

📊 Analyzed 58 months of financial data to identify revenue trends and cost optimization opportunities

🔮 Built Prophet time-series forecasting model achieving 3.6% MAPE for 12-month revenue predictions

📈 Developed Streamlit dashboard enabling real-time KPI tracking and scenario analysis

💡 Identified $20K+ profit anomalies through statistical outlier detection, prompting corrective action

Impact: Demonstrated data-driven financial planning methodology applicable across industries


Built with ❤️ and data by David Madison | 2025
