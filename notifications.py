import os
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from twilio.rest import Client

def send_twilio_message(to_phone_number: str, message: str) -> tuple:
    """
    Send SMS message using Twilio
    
    Parameters:
    -----------
    to_phone_number : str
        The phone number to send the SMS to (format: +1XXXXXXXXXX)
    message : str
        The message to send
    
    Returns:
    --------
    tuple
        (success, message) where success is a boolean indicating whether the operation was successful
        and message is a string with the result or error message
    """
    # Try to get credentials from session state first
    account_sid = None
    auth_token = None
    phone_number = None
    
    # Check if credentials are in session state
    if 'twilio_account_sid' in st.session_state and 'twilio_auth_token' in st.session_state and 'twilio_phone_number' in st.session_state:
        account_sid = st.session_state['twilio_account_sid']
        auth_token = st.session_state['twilio_auth_token']
        phone_number = st.session_state['twilio_phone_number']
    else:
        # Fallback to environment variables
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
    
    # Check if Twilio credentials are available
    if not account_sid or not auth_token or not phone_number:
        return (False, "Twilio credentials are not configured. Please configure your Twilio credentials.")
    
    try:
        client = Client(account_sid, auth_token)
        
        # Sending the SMS message
        message_obj = client.messages.create(
            body=message, 
            from_=phone_number, 
            to=to_phone_number
        )
        
        return (True, f"Message sent with SID: {message_obj.sid}")
    
    except Exception as e:
        return (False, f"Error sending message: {str(e)}")


def format_peak_time_alert(forecast_df, threshold=None, top_n=3):
    """
    Format peak time forecast as an alert message
    
    Parameters:
    -----------
    forecast_df : pandas.DataFrame
        Forecast results from Prophet model
    threshold : int or None
        Order volume threshold to trigger alert
    top_n : int
        Number of peak times to include
    
    Returns:
    --------
    str
        Formatted message with peak time information
    """
    # Get the forecast for the next 7 days
    next_week = forecast_df.iloc[-7:].copy()
    
    # If a threshold is provided, filter peaks above threshold
    if threshold:
        peaks = next_week[next_week['yhat'] > threshold].sort_values(by='yhat', ascending=False).head(top_n)
    else:
        # Otherwise, just get the top N peaks
        peaks = next_week.sort_values(by='yhat', ascending=False).head(top_n)
    
    # Format the message
    message = "🏨 HOTEL ORDER FORECAST ALERT 🏨\n\n"
    message += "Expected peak order times for the coming week:\n\n"
    
    for i, (_, row) in enumerate(peaks.iterrows()):
        date_str = row['ds'].strftime('%A, %b %d')
        forecast_value = int(row['yhat'])
        message += f"{i+1}. {date_str}: ~{forecast_value} orders\n"
    
    message += "\nThis forecast helps you prepare staffing and inventory in advance."
    
    return message


def format_inventory_alert(ingredient_forecast, threshold_pct=80):
    """
    Format inventory alert message based on ingredient forecast
    
    Parameters:
    -----------
    ingredient_forecast : dict
        Dictionary with forecasted ingredient quantities for each day
    threshold_pct : int
        Percentage threshold for alerting (e.g., 80% means alert when usage is >80% of typical stock)
    
    Returns:
    --------
    str
        Formatted message with inventory alert information
    """
    # Calculate total needed for each ingredient for the forecast period
    total_ingredients = {}
    for date, ingredients in ingredient_forecast.items():
        for ing, qty in ingredients.items():
            total_ingredients[ing] = total_ingredients.get(ing, 0) + qty
    
    # Sort by quantity needed
    sorted_ingredients = sorted(total_ingredients.items(), key=lambda x: x[1], reverse=True)
    top_ingredients = sorted_ingredients[:5]
    
    # Format the message
    message = "🥗 INGREDIENT INVENTORY ALERT 🥗\n\n"
    message += "Top 5 ingredients needed for the coming week:\n\n"
    
    for i, (ing, qty) in enumerate(top_ingredients):
        message += f"{i+1}. {ing}: {qty:.1f} units\n"
    
    message += "\nMake sure to stock up on these ingredients to meet demand."
    
    return message


def format_staffing_alert(staffing_results, date_filter=None):
    """
    Format staffing alert message
    
    Parameters:
    -----------
    staffing_results : list
        List of dictionaries with staffing recommendations for each day
    date_filter : datetime or None
        If provided, only include staffing for this specific date
    
    Returns:
    --------
    str
        Formatted message with staffing alert information
    """
    # Convert to DataFrame for easier filtering
    staff_df = pd.DataFrame(staffing_results)
    
    # Filter by date if specified
    if date_filter:
        tomorrow = date_filter.strftime('%Y-%m-%d')
        staff_df = staff_df[staff_df['date'].dt.strftime('%Y-%m-%d') == tomorrow]
    else:
        # Get next 3 days by default
        today = datetime.now()
        next_3_days = [(today + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, 4)]
        staff_df = staff_df[staff_df['date'].dt.strftime('%Y-%m-%d').isin(next_3_days)]
    
    # Format the message
    message = "👥 STAFFING REQUIREMENTS ALERT 👥\n\n"
    
    if len(staff_df) == 0:
        message += "No staffing data available for the requested period."
    else:
        for _, row in staff_df.iterrows():
            date_str = row['date'].strftime('%A, %b %d')
            message += f"Date: {date_str}\n"
            
            # Add each staff type and count
            for col in staff_df.columns:
                if col != 'date' and pd.notna(row[col]):
                    message += f"- {col}: {int(row[col])}\n"
            
            message += "\n"
    
    message += "Please adjust staffing schedules accordingly to ensure proper coverage during peak times."
    
    return message