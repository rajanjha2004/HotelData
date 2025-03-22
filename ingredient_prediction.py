import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def predict_ingredient_usage(df, forecast_df, ingredient_mapping, days_to_forecast=7):
    """
    Predict ingredient usage based on forecasted order volumes
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    forecast_df : pandas.DataFrame
        Forecast results from Prophet model
    ingredient_mapping : dict
        Mapping of menu items to their ingredients and quantities
    days_to_forecast : int
        Number of days to forecast ahead
    
    Returns:
    --------
    dict
        Dictionary with forecasted ingredient quantities for each day
    """
    # Get the forecasted order volume for the forecast period
    forecast_period = forecast_df.iloc[-days_to_forecast:].copy()
    
    # Calculate typical item distribution (percentage of each item in total orders)
    item_distribution = df.groupby('itemName')['itemQuantity'].sum() / df['itemQuantity'].sum()
    
    # Create a dictionary to store ingredient forecasts by day
    ingredient_forecast = {}
    
    # For each forecasted day
    for _, row in forecast_period.iterrows():
        date = row['ds'].strftime('%Y-%m-%d')
        forecasted_orders = row['yhat']  # Predicted total orders for the day
        
        # Initialize dictionary for this day's ingredients
        ingredient_forecast[date] = {}
        
        # For each menu item, calculate expected quantity and required ingredients
        for item, percentage in item_distribution.items():
            expected_quantity = forecasted_orders * percentage
            
            # If we have ingredient mapping for this item
            if item in ingredient_mapping:
                # For each ingredient in this item
                for ingredient, qty_per_item in ingredient_mapping[item].items():
                    # Calculate total ingredient quantity needed
                    ingredient_qty = expected_quantity * qty_per_item
                    
                    # Add to the day's ingredient forecast
                    if ingredient in ingredient_forecast[date]:
                        ingredient_forecast[date][ingredient] += ingredient_qty
                    else:
                        ingredient_forecast[date][ingredient] = ingredient_qty
        
        # Round the ingredient quantities to 2 decimal places
        for ingredient in ingredient_forecast[date]:
            ingredient_forecast[date][ingredient] = round(ingredient_forecast[date][ingredient], 2)
    
    return ingredient_forecast

def calculate_inventory_needs(ingredient_forecast, current_inventory={}, reorder_threshold={}):
    """
    Calculate inventory needs based on ingredient forecast and current inventory levels
    
    Parameters:
    -----------
    ingredient_forecast : dict
        Dictionary with forecasted ingredient quantities for each day
    current_inventory : dict
        Dictionary with current inventory levels for each ingredient
    reorder_threshold : dict
        Dictionary with reorder thresholds for each ingredient
    
    Returns:
    --------
    dict
        Dictionary with inventory needs and reorder recommendations
    """
    # Initialize inventory needs dictionary
    inventory_needs = {
        'total_needed': {},
        'reorder_recommendations': {}
    }
    
    # Calculate total needed for each ingredient across all forecasted days
    for date, ingredients in ingredient_forecast.items():
        for ingredient, qty in ingredients.items():
            if ingredient in inventory_needs['total_needed']:
                inventory_needs['total_needed'][ingredient] += qty
            else:
                inventory_needs['total_needed'][ingredient] = qty
    
    # Generate reorder recommendations
    for ingredient, needed_qty in inventory_needs['total_needed'].items():
        current_qty = current_inventory.get(ingredient, 0)
        threshold = reorder_threshold.get(ingredient, needed_qty * 0.2)  # Default threshold is 20% of needed quantity
        
        if current_qty < needed_qty:
            inventory_needs['reorder_recommendations'][ingredient] = {
                'current_inventory': current_qty,
                'needed_quantity': needed_qty,
                'deficit': needed_qty - current_qty,
                'reorder_suggested': current_qty < threshold
            }
    
    return inventory_needs
