import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_order_volume_by_hour(df):
    """
    Create a plot showing order volume by hour of day
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the visualization
    """
    # Group by hour and count orders
    hourly_orders = df.groupby('order_hour').size().reset_index()
    hourly_orders.columns = ['Hour', 'Order Count']
    
    # Create the plot
    fig = px.bar(
        hourly_orders, 
        x='Hour', 
        y='Order Count',
        title='Order Volume by Hour of Day',
        labels={'Hour': 'Hour of Day (24h)', 'Order Count': 'Number of Orders'},
        text='Order Count'
    )
    
    # Highlight peak hours
    peak_hours = hourly_orders.sort_values('Order Count', ascending=False)['Hour'].head(3).tolist()
    
    for hour in peak_hours:
        fig.add_annotation(
            x=hour,
            y=hourly_orders[hourly_orders['Hour'] == hour]['Order Count'].values[0],
            text="Peak",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    # Customize layout
    fig.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        bargap=0.1
    )
    
    return fig

def plot_order_volume_by_day(df):
    """
    Create a plot showing order volume by day of week
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Preprocessed hotel order data
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the visualization
    """
    # Group by day of week and count orders
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_orders = df.groupby('order_day').size().reset_index()
    daily_orders.columns = ['Day', 'Order Count']
    
    # Ensure proper day ordering
    daily_orders['Day'] = pd.Categorical(daily_orders['Day'], categories=day_order, ordered=True)
    daily_orders = daily_orders.sort_values('Day')
    
    # Create the plot
    fig = px.bar(
        daily_orders, 
        x='Day', 
        y='Order Count',
        title='Order Volume by Day of Week',
        labels={'Day': 'Day of Week', 'Order Count': 'Number of Orders'},
        text='Order Count',
        color='Order Count',
        color_continuous_scale='Viridis'
    )
    
    # Highlight peak days
    peak_days = daily_orders.sort_values('Order Count', ascending=False)['Day'].head(2).tolist()
    
    for day in peak_days:
        fig.add_annotation(
            x=day,
            y=daily_orders[daily_orders['Day'] == day]['Order Count'].values[0],
            text="Peak",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    return fig

def plot_ingredient_forecast(ingredient_forecast):
    """
    Create a plot showing ingredient usage forecast
    
    Parameters:
    -----------
    ingredient_forecast : dict
        Dictionary with forecasted ingredient quantities for each day
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the visualization
    """
    # Convert the forecast dictionary to a dataframe
    data = []
    for date, ingredients in ingredient_forecast.items():
        for ingredient, quantity in ingredients.items():
            data.append({
                'Date': date,
                'Ingredient': ingredient,
                'Quantity': quantity
            })
    
    df = pd.DataFrame(data)
    
    # Get top 5 ingredients by total quantity
    top_ingredients = df.groupby('Ingredient')['Quantity'].sum().sort_values(ascending=False).head(5).index.tolist()
    
    # Filter dataframe to include only top ingredients
    df_top = df[df['Ingredient'].isin(top_ingredients)]
    
    # Create the plot
    fig = px.line(
        df_top,
        x='Date',
        y='Quantity',
        color='Ingredient',
        title='Forecasted Usage of Top 5 Ingredients',
        labels={'Date': 'Date', 'Quantity': 'Quantity Needed', 'Ingredient': 'Ingredient'},
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Quantity Needed',
        legend_title='Ingredient',
        hovermode='x unified'
    )
    
    return fig

def plot_staff_requirements(staffing_results):
    """
    Create a plot showing staffing requirements forecast
    
    Parameters:
    -----------
    staffing_results : list
        List of dictionaries with staffing recommendations for each day
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the visualization
    """
    # Handle empty or invalid data
    if not staffing_results:
        fig = go.Figure()
        fig.add_annotation(
            text="No staffing data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Convert the staffing results to a dataframe
    df = pd.DataFrame(staffing_results)
    
    # Check if the 'date' column is present
    if 'date' not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="Missing 'date' column in staffing data",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Get staff types (excluding non-staff columns)
    staff_columns = []
    for col in df.columns:
        if col not in ['date', 'predicted_orders', 'lower_bound', 'upper_bound', 'total_staff']:
            staff_columns.append(col)
    
    # Create a figure
    fig = go.Figure()
    
    # Add trace for total staff
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['total_staff'],
        mode='lines+markers',
        name='Total Staff',
        line=dict(width=3, dash='dot'),
        marker=dict(size=10)
    ))
    
    # Add trace for each staff type
    for staff_type in staff_columns:
        fig.add_trace(go.Bar(
            x=df['date'],
            y=df[staff_type],
            name=staff_type
        ))
    
    # Add range for order predictions
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['upper_bound'] / 20,  # Scale down orders to fit on same scale as staff
        mode='lines',
        name='Orders (Upper Bound)',
        line=dict(width=0),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['lower_bound'] / 20,  # Scale down orders to fit on same scale as staff
        mode='lines',
        name='Order Range',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(0, 0, 255, 0.1)'
    ))
    
    # Customize layout
    fig.update_layout(
        title='Staffing Requirements Forecast',
        xaxis_title='Date',
        yaxis_title='Number of Staff',
        barmode='stack',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def plot_revenue_trends(revenue_df):
    """
    Create a plot showing revenue trends
    
    Parameters:
    -----------
    revenue_df : pandas.DataFrame
        DataFrame with revenue data by date
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with the visualization
    """
    # Create the plot
    fig = px.line(
        revenue_df,
        x='createdAt',
        y='total_price',
        title='Daily Revenue Trend',
        labels={'createdAt': 'Date', 'total_price': 'Revenue ($)'},
        markers=True
    )
    
    # Add 7-day moving average
    revenue_df['7day_ma'] = revenue_df['total_price'].rolling(window=7, min_periods=1).mean()
    
    fig.add_trace(go.Scatter(
        x=revenue_df['createdAt'],
        y=revenue_df['7day_ma'],
        mode='lines',
        name='7-Day Moving Average',
        line=dict(color='red', width=3)
    ))
    
    # Customize layout
    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Revenue ($)',
        hovermode='x unified'
    )
    
    return fig
