import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import re

def generate_sample_data(num_orders=1000, start_date=None, end_date=None):
    """
    Generate sample hotel order data for testing
    Note: This function is for development/testing only and should be removed in production
    
    Parameters:
    -----------
    num_orders : int
        Number of sample orders to generate
    start_date : str
        Start date for the sample data (format: 'YYYY-MM-DD')
    end_date : str
        End date for the sample data (format: 'YYYY-MM-DD')
    
    Returns:
    --------
    pandas.DataFrame
        Sample order data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Convert to datetime
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Random order items and prices
    menu_items = {
        'Burger and Fries': 15.99,
        'Pasta Carbonara': 18.99,
        'Grilled Salmon': 24.99,
        'Caesar Salad': 12.99,
        'Club Sandwich': 14.99,
        'Steak Dinner': 29.99,
        'Vegetable Stir Fry': 16.99,
        'Seafood Platter': 32.99,
        'Chicken Curry': 19.99,
        'Pizza Margherita': 17.99,
        'Mushroom Risotto': 20.99,
        'Fish and Chips': 18.99,
        'Beef Lasagna': 21.99,
        'Shrimp Scampi': 25.99,
        'Chocolate Cake': 8.99,
        'Cheesecake': 9.99,
        'Ice Cream Sundae': 7.99,
        'Coffee': 3.99,
        'Soft Drink': 2.99,
        'Glass of Wine': 10.99
    }
    
    # Generate random data
    orders = []
    for i in range(1, num_orders + 1):
        order_id = f"ORD-{i:06d}"
        hotel_id = np.random.randint(1, 6)  # 5 different hotels
        order_no = f"ON-{i:06d}"
        
        # Generate a random timestamp between start and end date
        time_range = (end_date - start_date).total_seconds()
        random_seconds = np.random.randint(0, time_range)
        created_at = start_date + timedelta(seconds=random_seconds)
        
        # Add hourly and day of week patterns
        # More orders during meal times and weekends
        hour = created_at.hour
        day_of_week = created_at.dayofweek
        
        # Skip this order with higher probability during non-peak hours
        if hour < 7 or hour > 22:  # Late night/early morning
            if np.random.random() < 0.9:  # 90% chance to skip
                continue
        elif hour not in [7, 8, 12, 13, 18, 19, 20]:  # Not meal times
            if np.random.random() < 0.6:  # 60% chance to skip
                continue
        
        # More orders on weekends (days 5 and 6)
        if day_of_week < 5 and np.random.random() < 0.3:  # 30% chance to skip on weekdays
            continue
        
        # Generate 1-5 items per order
        num_items = np.random.randint(1, 6)
        for j in range(num_items):
            item_name = np.random.choice(list(menu_items.keys()))
            item_price = menu_items[item_name]
            item_quantity = np.random.randint(1, 4)  # 1-3 of each item
            
            # Order status (mostly completed)
            status_options = ['completed', 'pending', 'canceled']
            status_weights = [0.85, 0.1, 0.05]  # 85% completed, 10% pending, 5% canceled
            status = np.random.choice(status_options, p=status_weights)
            
            # Updated timestamp (after created_at)
            process_time = np.random.randint(10, 60)  # 10-60 minutes to process
            updated_at = created_at + timedelta(minutes=process_time)
            
            orders.append({
                'orderId': order_id,
                'hotelId': hotel_id,
                'orderNo': order_no,
                'itemName': item_name,
                'itemQuantity': item_quantity,
                'itemPrice': item_price,
                'status': status,
                'createdAt': created_at,
                'updatedAt': updated_at
            })
    
    # Create DataFrame
    df = pd.DataFrame(orders)
    
    return df

def extract_ingredients_from_item_name(item_name):
    """
    Extract potential ingredients from an item name using simple heuristics
    
    Parameters:
    -----------
    item_name : str
        Name of the menu item
    
    Returns:
    --------
    list
        List of potential ingredients
    """
    # Remove common words
    common_words = ['and', 'with', 'the', 'a', 'of', 'in', 'on', 'for']
    words = item_name.lower().split()
    filtered_words = [word for word in words if word not in common_words]
    
    # Remove special characters
    filtered_words = [re.sub(r'[^a-zA-Z]', '', word) for word in filtered_words]
    
    # Remove empty strings
    filtered_words = [word for word in filtered_words if word]
    
    return filtered_words

def calculate_order_processing_metrics(df):
    """
    Calculate order processing metrics
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    
    Returns:
    --------
    dict
        Dictionary with order processing metrics
    """
    metrics = {}
    
    # Average processing time
    completed_orders = df[df['status'].str.lower() == 'completed']
    metrics['avg_processing_time'] = completed_orders['processing_time'].mean()
    
    # Orders by status
    status_counts = df.groupby('status').size()
    metrics['status_distribution'] = status_counts.to_dict()
    
    # Average order value
    df['order_value'] = df['itemPrice'] * df['itemQuantity']
    metrics['avg_order_value'] = df.groupby('orderId')['order_value'].sum().mean()
    
    # Average items per order
    metrics['avg_items_per_order'] = df.groupby('orderId')['itemQuantity'].sum().mean()
    
    return metrics
