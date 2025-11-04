"""
Financial Performance Dashboard
Interactive Streamlit Application

A clean, simple dashboard for monitoring financial KPIs and forecasts.
Author: David Madison
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean design
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    h1 {
        color: #1f1f1f;
        font-weight: 700;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load financial data"""
    df = pd.read_csv('data/financials.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data
def forecast_data(df, column, periods=12):
    """Generate forecast for specified column"""
    prophet_df = df[['Date', column]].copy()
    prophet_df.columns = ['ds', 'y']
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(prophet_df)
    
    future = model.make_future_dataframe(periods=periods, freq='MS')
    forecast = model.predict(future)
    
    return forecast

def main():
    """Main dashboard function"""
    
    # Header
    st.title("💼 Financial Performance Dashboard")
    st.markdown("Real-time financial analytics and forecasting for data-driven decision-making")
    st.markdown("---")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.title("📊 Dashboard Controls")
    
    # Date range selector
    date_range = st.sidebar.selectbox(
        "Time Period",
        ["All Time", "Last 12 Months", "Last 6 Months", "Last Quarter"]
    )
    
    if date_range == "Last 12 Months":
        df_filtered = df.tail(12)
    elif date_range == "Last 6 Months":
        df_filtered = df.tail(6)
    elif date_range == "Last Quarter":
        df_filtered = df.tail(3)
    else:
        df_filtered = df
    
    # Forecast periods
    forecast_periods = st.sidebar.slider(
        "Forecast Months",
        min_value=3,
        max_value=12,
        value=6,
        step=1
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This dashboard analyzes financial performance and generates forecasts using Prophet time-series modeling.")
    
    # Calculate KPIs
    total_revenue = df_filtered['Revenue'].sum()
    total_expenses = df_filtered['Expenses'].sum()
    total_profit = df_filtered['Net_Profit'].sum()
    avg_margin = df_filtered['Profit_Margin'].mean()
    
    # Calculate changes
    if len(df_filtered) > 1:
        revenue_change = ((df_filtered['Revenue'].iloc[-1] / df_filtered['Revenue'].iloc[0]) - 1) * 100
        profit_change = ((df_filtered['Net_Profit'].iloc[-1] / df_filtered['Net_Profit'].iloc[0]) - 1) * 100
    else:
        revenue_change = 0
        profit_change = 0
    
    # KPI Cards
    st.subheader("📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{revenue_change:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Total Expenses",
            value=f"${total_expenses:,.0f}",
            delta=f"{((df_filtered['Expenses'].iloc[-1] / df_filtered['Expenses'].iloc[0]) - 1) * 100:.1f}%" if len(df_filtered) > 1 else "0%"
        )
    
    with col3:
        st.metric(
            label="Net Profit",
            value=f"${total_profit:,.0f}",
            delta=f"{profit_change:.1f}%"
        )
    
    with col4:
        st.metric(
            label="Avg Profit Margin",
            value=f"{avg_margin:.1f}%",
            delta=f"{(df_filtered['Profit_Margin'].iloc[-1] - avg_margin):.1f}%" if len(df_filtered) > 1 else "0%"
        )
    
    st.markdown("---")
    
    # Revenue vs Expenses Chart
    st.subheader("💰 Revenue vs Expenses")
    
    fig_rev_exp = go.Figure()
    
    fig_rev_exp.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Revenue'],
        mode='lines+markers',
        name='Revenue',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=6)
    ))
    
    fig_rev_exp.add_trace(go.Scatter(
        x=df_filtered['Date'],
        y=df_filtered['Expenses'],
        mode='lines+markers',
        name='Expenses',
        line=dict(color='#C62828', width=3),
        marker=dict(size=6)
    ))
    
    fig_rev_exp.update_layout(
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_rev_exp, use_container_width=True)
    
    # Two column layout for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Profit Margin Trend")
        
        fig_margin = go.Figure()
        
        fig_margin.add_trace(go.Scatter(
            x=df_filtered['Date'],
            y=df_filtered['Profit_Margin'],
            mode='lines+markers',
            name='Profit Margin',
            line=dict(color='#1565C0', width=3),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor='rgba(21, 101, 192, 0.1)'
        ))
        
        fig_margin.add_hline(
            y=avg_margin,
            line_dash="dash",
            line_color="#FF6F00",
            annotation_text=f"Average: {avg_margin:.1f}%",
            annotation_position="right"
        )
        
        fig_margin.update_layout(
            xaxis_title="Date",
            yaxis_title="Profit Margin (%)",
            hovermode='x unified',
            plot_bgcolor='white',
            height=350
        )
        
        st.plotly_chart(fig_margin, use_container_width=True)
    
    with col2:
        st.subheader("💸 Expense Breakdown")
        
        avg_marketing = df_filtered['Marketing_Cost'].mean()
        avg_operational = df_filtered['Operational_Cost'].mean()
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Marketing', 'Operational'],
            values=[avg_marketing, avg_operational],
            hole=.4,
            marker_colors=['#FF6F00', '#1565C0']
        )])
        
        fig_pie.update_layout(
            height=350,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("---")
    
    # Forecasting Section
    st.subheader("🔮 Revenue Forecast")
    
    with st.spinner('Generating forecast...'):
        revenue_forecast = forecast_data(df, 'Revenue', periods=forecast_periods)
    
    # Combine historical and forecast
    fig_forecast = go.Figure()
    
    # Historical data
    fig_forecast.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Revenue'],
        mode='lines+markers',
        name='Historical',
        line=dict(color='#2E7D32', width=3),
        marker=dict(size=6)
    ))
    
    # Forecast
    forecast_future = revenue_forecast.tail(forecast_periods)
    
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future['ds'],
        y=forecast_future['yhat'],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#1565C0', width=3, dash='dash'),
        marker=dict(size=6, symbol='diamond')
    ))
    
    # Confidence interval
    fig_forecast.add_trace(go.Scatter(
        x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
        y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(21, 101, 192, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence',
        showlegend=True
    ))
    
    fig_forecast.update_layout(
        xaxis_title="Date",
        yaxis_title="Revenue ($)",
        hovermode='x unified',
        plot_bgcolor='white',
        height=450,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    # Forecast summary
    forecasted_revenue = forecast_future['yhat'].sum()
    current_revenue = df.tail(forecast_periods)['Revenue'].sum()
    forecast_growth = ((forecasted_revenue - current_revenue) / current_revenue) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"Forecasted Revenue ({forecast_periods} months)",
            value=f"${forecasted_revenue:,.0f}",
            delta=f"{forecast_growth:.1f}% vs last {forecast_periods} months"
        )
    
    with col2:
        st.metric(
            label="Expected Monthly Average",
            value=f"${forecasted_revenue / forecast_periods:,.0f}"
        )
    
    with col3:
        lower_bound = forecast_future['yhat_lower'].sum()
        upper_bound = forecast_future['yhat_upper'].sum()
        st.metric(
            label="Confidence Range",
            value=f"${lower_bound:,.0f} - ${upper_bound:,.0f}"
        )
    
    st.markdown("---")
    
    # Data table
    with st.expander("📋 View Raw Data"):
        st.dataframe(
            df_filtered[['Date', 'Revenue', 'Expenses', 'Net_Profit', 'Profit_Margin']].sort_values('Date', ascending=False),
            use_container_width=True
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Financial Performance Dashboard | Built with Streamlit & Prophet</p>
            <p>Data Analytics Portfolio Project by David Madison</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()