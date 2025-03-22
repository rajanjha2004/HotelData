import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def optimize_staffing(forecast_df, orders_per_staff=5, min_staff=2, prep_time_factor=1.0, staff_types=None):
    """
    Optimize staffing needs based on forecasted order volumes
    
    Parameters:
    -----------
    forecast_df : pandas.DataFrame
        Forecast results from Prophet model
    orders_per_staff : int
        Average number of orders a staff member can handle per hour
    min_staff : int
        Minimum number of staff required at all times
    prep_time_factor : float
        Adjustment factor for preparation time
    staff_types : list
        List of staff types/roles to consider
    
    Returns:
    --------
    list
        List of dictionaries with staffing recommendations for each day
    """
    if staff_types is None:
        staff_types = ["Chefs", "Waiters", "Kitchen helpers"]
    
    # Use only the forecast part of the dataframe
    forecast_period = forecast_df[forecast_df['ds'] > datetime.now()].copy()
    
    # Calculate staff needed per day
    staffing_results = []
    
    for _, row in forecast_period.iterrows():
        date = row['ds']
        predicted_orders = max(row['yhat'], 0)  # Ensure non-negative
        
        # Get confidence interval values
        lower_bound = max(row['yhat_lower'], 0)
        upper_bound = max(row['yhat_upper'], 0)
        
        # Adjust by prep time factor
        adjusted_orders = predicted_orders * prep_time_factor
        
        # Calculate total staff needed based on orders per staff
        # We assume orders are distributed within 12 hours of operation
        hourly_order_rate = adjusted_orders / 12  # Divide by 12 hours of operation
        total_staff_needed = max(np.ceil(hourly_order_rate / orders_per_staff), min_staff)
        
        # Create staffing distribution object
        staffing_day = {
            'date': date,
            'predicted_orders': int(predicted_orders),
            'lower_bound': int(lower_bound),
            'upper_bound': int(upper_bound),
            'total_staff': int(total_staff_needed)
        }
        
        # Distribute staff across types
        # This is a simplified distribution; a real implementation could be more sophisticated
        if "Chefs" in staff_types:
            staffing_day["Chefs"] = max(int(total_staff_needed * 0.35), 1)
        
        if "Waiters" in staff_types:
            staffing_day["Waiters"] = max(int(total_staff_needed * 0.4), 1)
        
        if "Kitchen helpers" in staff_types:
            staffing_day["Kitchen helpers"] = max(int(total_staff_needed * 0.15), 1)
        
        if "Bartenders" in staff_types:
            staffing_day["Bartenders"] = max(int(total_staff_needed * 0.1), 1)
        
        # Add to results
        staffing_results.append(staffing_day)
    
    return staffing_results

def calculate_staffing_costs(staffing_results, hourly_rates, shift_hours=8):
    """
    Calculate staffing costs based on staffing recommendations
    
    Parameters:
    -----------
    staffing_results : list
        List of dictionaries with staffing recommendations for each day
    hourly_rates : dict
        Dictionary with hourly rates for each staff type
    shift_hours : int
        Number of hours in a single shift
    
    Returns:
    --------
    dict
        Dictionary with staffing costs by day and staff type
    """
    # Initialize cost dictionary
    staffing_costs = {
        'daily_costs': [],
        'total_cost': 0,
        'cost_by_type': {}
    }
    
    # Calculate costs for each day
    for day in staffing_results:
        daily_cost = 0
        date = day['date']
        
        day_cost = {
            'date': date,
            'costs': {}
        }
        
        # Calculate cost for each staff type
        for staff_type, count in day.items():
            if staff_type in hourly_rates:
                type_cost = count * hourly_rates[staff_type] * shift_hours
                day_cost['costs'][staff_type] = type_cost
                daily_cost += type_cost
                
                # Add to total by type
                if staff_type in staffing_costs['cost_by_type']:
                    staffing_costs['cost_by_type'][staff_type] += type_cost
                else:
                    staffing_costs['cost_by_type'][staff_type] = type_cost
        
        day_cost['total'] = daily_cost
        staffing_costs['daily_costs'].append(day_cost)
        staffing_costs['total_cost'] += daily_cost
    
    return staffing_costs
