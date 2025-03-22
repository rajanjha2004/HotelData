import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(df):
    """
    Load and preprocess the hotel order data
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw hotel order data
    
    Returns:
    --------
    pandas.DataFrame
        Preprocessed data
    """
    # Make a copy to avoid modifying original data
    processed_df = df.copy()
    
    # Convert timestamp strings to datetime
    processed_df['createdAt'] = pd.to_datetime(processed_df['createdAt'])
    processed_df['updatedAt'] = pd.to_datetime(processed_df['updatedAt'])
    
    # Create additional time-related features
    processed_df['order_hour'] = processed_df['createdAt'].dt.hour
    processed_df['order_day'] = processed_df['createdAt'].dt.day_name()
    processed_df['order_date'] = processed_df['createdAt'].dt.date
    processed_df['order_month'] = processed_df['createdAt'].dt.month
    processed_df['order_year'] = processed_df['createdAt'].dt.year
    processed_df['is_weekend'] = processed_df['createdAt'].dt.dayofweek >= 5
    
    # Calculate order processing time in minutes
    processed_df['processing_time'] = (processed_df['updatedAt'] - processed_df['createdAt']).dt.total_seconds() / 60
    
    # Handle missing values
    processed_df['itemPrice'] = processed_df['itemPrice'].fillna(0)
    processed_df['itemQuantity'] = processed_df['itemQuantity'].fillna(0)
    
    # Filter out invalid entries
    processed_df = processed_df[processed_df['itemQuantity'] > 0]
    processed_df = processed_df[processed_df['itemPrice'] >= 0]
    
    # Filter out entries with future timestamps
    current_date = pd.Timestamp.now().tz_localize(None)
    # Convert createdAt to timezone-naive to ensure proper comparison
    processed_df = processed_df[processed_df['createdAt'].dt.tz_localize(None).dt.tz_localize(None) <= current_date]
    
    # Calculate total price for each item
    processed_df['total_price'] = processed_df['itemPrice'] * processed_df['itemQuantity']
    
    # Handle orders with status information
    processed_df['is_completed'] = processed_df['status'].str.lower() == 'completed'
    processed_df['is_canceled'] = processed_df['status'].str.lower() == 'canceled'
    
    return processed_df
