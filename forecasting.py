import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import datetime, timedelta

def forecast_peak_times(df, days_to_forecast=7, confidence_interval=90):
    """
    Forecast peak order times using Facebook Prophet
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    days_to_forecast : int
        Number of days to forecast ahead
    confidence_interval : int
        Confidence interval for the forecast (in percentage)
    
    Returns:
    --------
    pandas.DataFrame
        Forecast results including predictions and confidence intervals
    """
    # Prepare data for Prophet
    # Group by date and count orders
    # First, make a copy and ensure timezone is removed (Prophet doesn't support timezones)
    data = df.copy()
    if hasattr(data['createdAt'].dt, 'tz') and data['createdAt'].dt.tz is not None:
        data['createdAt'] = data['createdAt'].dt.tz_localize(None)
        
    data = data.set_index('createdAt')
    daily_orders = data.resample('D').size().reset_index()
    daily_orders.columns = ['ds', 'y']
    
    # Ensure the ds column has no timezone
    if hasattr(daily_orders['ds'].dt, 'tz') and daily_orders['ds'].dt.tz is not None:
        daily_orders['ds'] = daily_orders['ds'].dt.tz_localize(None)
    
    # Create and train Prophet model
    model = Prophet(
        interval_width=confidence_interval/100,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'  # Use multiplicative mode for retail data
    )
    
    # Add custom seasonality for hotel data - higher order volumes during weekends and holidays
    model.add_seasonality(
        name='weekend_effect',
        period=7,
        fourier_order=3,
        condition_name='is_weekend'
    )
    
    # Add weekend indicator
    daily_orders['is_weekend'] = (daily_orders['ds'].dt.dayofweek >= 5).astype(int)
    
    # Fit the model
    model.fit(daily_orders)
    
    # Make future dataframe
    future = model.make_future_dataframe(periods=days_to_forecast)
    future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)
    
    # Predict
    forecast = model.predict(future)
    
    return forecast

def forecast_hourly_peaks(df, days_to_forecast=7):
    """
    Forecast hourly peak times to identify busy hours of the day
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    days_to_forecast : int
        Number of days to forecast ahead
    
    Returns:
    --------
    dict
        Dictionary with forecasted peak hours for each day
    """
    # Group data by hour of day
    hourly_data = df.groupby('order_hour').size().reset_index()
    hourly_data.columns = ['hour', 'order_count']
    
    # Find the typical peak hours
    peak_hours = hourly_data.sort_values('order_count', ascending=False)['hour'].tolist()[:3]
    
    # Create a dictionary to store peak hours for each forecasted day
    peak_hour_forecast = {}
    
    # For each forecasted day, assign the typical peak hours
    # In a real implementation, this could use a more sophisticated time series model
    # that accounts for day of week variations
    for i in range(days_to_forecast):
        forecast_date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
        peak_hour_forecast[forecast_date] = peak_hours
    
    return peak_hour_forecast
