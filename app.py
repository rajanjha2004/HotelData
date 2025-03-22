import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Import custom modules
from data_preprocessing import load_and_preprocess_data
from forecasting import forecast_peak_times
from ingredient_prediction import predict_ingredient_usage
from staffing_optimization import optimize_staffing
from visualization import (
    plot_order_volume_by_hour,
    plot_order_volume_by_day,
    plot_ingredient_forecast,
    plot_staff_requirements,
    plot_revenue_trends
)
from utils import generate_sample_data
from notifications import (
    send_twilio_message, 
    format_peak_time_alert,
    format_inventory_alert,
    format_staffing_alert
)

# Set page configuration
st.set_page_config(
    page_title="Hotel Order Analysis System",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title
st.title("🏨 Hotel Order Analysis System")
st.write("Analyze hotel order data, forecast peak times, predict ingredient usage, and optimize staffing needs.")

# We removed SMS notifications functionality

# Sidebar for uploading data and configuring models
with st.sidebar:
    st.header("Data Input & Configuration")
    
    st.markdown("### Data Source")
    data_source = st.radio("Choose a data source", ["Upload a CSV file", "Use sample data for testing"])
    
    if data_source == "Upload a CSV file":
        uploaded_file = st.file_uploader("Upload CSV file with order data", type=["csv"])
        
        if uploaded_file is not None:
            # Load the data
            try:
                df_raw = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded data with {len(df_raw)} rows.")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                df_raw = None
        else:
            st.info("Please upload order data CSV file to begin analysis.")
            st.markdown("""
            ### Expected CSV format:
            The CSV should contain the following columns:
            - `orderId`: Unique identifier for each order
            - `hotelId`: Hotel identification number
            - `orderNo`: Order number
            - `itemName`: Name of the ordered item
            - `itemQuantity`: Quantity of the ordered item
            - `itemPrice`: Price of the ordered item
            - `status`: Current status of the order (e.g., completed, pending, canceled)
            - `createdAt`: Timestamp when the order was placed
            - `updatedAt`: Timestamp when the order was last updated
            """)
            df_raw = None
    else:
        # Generate sample data
        num_orders = st.slider("Number of sample orders", min_value=100, max_value=5000, value=1000, step=100)
        
        # Date range for sample data
        st.markdown("#### Date Range for Sample Data")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=90))
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
        
        if st.button("Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                df_raw = generate_sample_data(
                    num_orders=num_orders, 
                    start_date=start_date.strftime("%Y-%m-%d"), 
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                st.success(f"Successfully generated sample data with {len(df_raw)} rows.")
        else:
            st.info("Click the 'Generate Sample Data' button to create test data.")
            df_raw = None
    
    st.header("Analysis Configuration")
    
    # Forecast horizon
    forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)
    
    # Time granularity for analysis
    time_granularity = st.selectbox(
        "Time granularity for analysis",
        options=["Hourly", "Daily", "Weekly"],
        index=0
    )
    
    # Confidence interval
    confidence_interval = st.slider("Confidence interval (%)", min_value=80, max_value=95, value=90, step=5)

# Main content
if df_raw is not None:
    try:
        # Data preprocessing
        with st.spinner("Preprocessing data..."):
            df = load_and_preprocess_data(df_raw)
            st.session_state['preprocessed_data'] = df
        
        # Display tabs for different analyses - focusing on core features
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Data Overview", 
            "⏱️ Peak Time Forecast", 
            "🥗 Ingredient Prediction", 
            "👥 Staffing Optimization"
        ])
        
        with tab1:
            st.header("Data Overview")
            
            # Display data statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Sample Data")
                st.dataframe(df.head())
            
            with col2:
                st.subheader("Data Statistics")
                st.write(f"Total Orders: {df['orderId'].nunique()}")
                st.write(f"Total Items Ordered: {df['itemQuantity'].sum()}")
                st.write(f"Date Range: {df['createdAt'].min()} to {df['createdAt'].max()}")
                st.write(f"Number of Different Items: {df['itemName'].nunique()}")
                
            # Overall order volume trend
            st.subheader("Order Volume Trend")
            df_daily = df.set_index('createdAt').resample('D')['itemQuantity'].sum().reset_index()
            fig = px.line(
                df_daily, 
                x='createdAt', 
                y='itemQuantity', 
                title='Daily Order Volume',
                labels={'createdAt': 'Date', 'itemQuantity': 'Number of Items Ordered'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Top ordered items
            st.subheader("Top Ordered Items")
            top_items = df.groupby('itemName')['itemQuantity'].sum().sort_values(ascending=False).head(10)
            fig = px.bar(
                top_items, 
                x=top_items.index, 
                y=top_items.values,
                labels={'x': 'Item Name', 'y': 'Total Quantity Ordered'},
                title='Top 10 Most Ordered Items'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab2:
            st.header("Peak Time Forecast")
            
            # Current peak hours
            st.subheader("Current Peak Hours")
            hourly_plot = plot_order_volume_by_hour(df)
            st.plotly_chart(hourly_plot, use_container_width=True)
            
            # Current peak days
            st.subheader("Current Peak Days")
            daily_plot = plot_order_volume_by_day(df)
            st.plotly_chart(daily_plot, use_container_width=True)
            
            # Forecast future peak times
            st.subheader(f"Forecasted Peak Times (Next {forecast_days} Days)")
            with st.spinner("Generating forecast..."):
                forecast_result = forecast_peak_times(df, forecast_days, confidence_interval)
                
                # Plot forecast results
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'][:len(forecast_result)-forecast_days], 
                    y=forecast_result['y'][:len(forecast_result)-forecast_days],
                    mode='lines',
                    name='Historical'
                ))
                
                # Add forecasted data
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'][len(forecast_result)-forecast_days:], 
                    y=forecast_result['yhat'][len(forecast_result)-forecast_days:],
                    mode='lines',
                    name='Forecast',
                    line=dict(color='red')
                ))
                
                # Add confidence interval
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'][len(forecast_result)-forecast_days:],
                    y=forecast_result['yhat_upper'][len(forecast_result)-forecast_days:],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=forecast_result['ds'][len(forecast_result)-forecast_days:],
                    y=forecast_result['yhat_lower'][len(forecast_result)-forecast_days:],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 0, 0, 0.2)',
                    name=f'{confidence_interval}% Confidence Interval'
                ))
                
                fig.update_layout(
                    title=f'Order Volume Forecast for Next {forecast_days} Days',
                    xaxis_title='Date',
                    yaxis_title='Order Volume',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display expected peak times
                peak_forecast = forecast_result[len(forecast_result)-forecast_days:].sort_values(by='yhat', ascending=False).head(3)
                
                st.subheader("Expected Peak Times")
                for i, (_, row) in enumerate(peak_forecast.iterrows()):
                    peak_date = row['ds'].strftime('%Y-%m-%d')
                    peak_value = int(row['yhat'])
                    st.write(f"**Peak {i+1}**: {peak_date} with ~{peak_value} expected orders")
        
        with tab3:
            st.header("Ingredient Prediction")
            
            # Create a mock mapping of items to ingredients
            if 'ingredient_mapping' not in st.session_state:
                unique_items = df['itemName'].unique()
                # This is a simplified example - in a real application,
                # this would be provided by the user or from a database
                st.session_state['ingredient_mapping'] = {}
                common_ingredients = ['Flour', 'Sugar', 'Eggs', 'Milk', 'Butter', 
                                    'Chicken', 'Beef', 'Vegetables', 'Cheese', 'Rice', 
                                    'Pasta', 'Tomatoes', 'Onions', 'Garlic', 'Oil']
                
                import random
                
                for item in unique_items:
                    # Randomly assign 2-5 ingredients to each item
                    num_ingredients = random.randint(2, 5)
                    selected_ingredients = random.sample(common_ingredients, num_ingredients)
                    # Assign random quantities
                    ingredient_quantities = {ing: round(random.uniform(0.1, 2.0), 2) for ing in selected_ingredients}
                    st.session_state['ingredient_mapping'][item] = ingredient_quantities
            
            # UI for ingredient mapping
            st.subheader("Item to Ingredient Mapping")
            st.write("This mapping shows how menu items relate to ingredients (for demonstration purposes).")
            
            # Select an item to show its ingredients
            selected_item = st.selectbox("Select an item to view ingredients", options=list(st.session_state['ingredient_mapping'].keys()))
            
            if selected_item:
                st.write(f"Ingredients for '{selected_item}':")
                ingredients_df = pd.DataFrame(
                    [(ing, qty) for ing, qty in st.session_state['ingredient_mapping'][selected_item].items()],
                    columns=['Ingredient', 'Quantity (units)']
                )
                st.dataframe(ingredients_df)
            
            # Predict future ingredient usage
            st.subheader("Predicted Ingredient Usage")
            with st.spinner("Calculating ingredient predictions..."):
                ingredient_forecast = predict_ingredient_usage(
                    df, 
                    forecast_result, 
                    st.session_state['ingredient_mapping'],
                    forecast_days
                )
                
                # Plot ingredient forecast
                ingredient_chart = plot_ingredient_forecast(ingredient_forecast)
                st.plotly_chart(ingredient_chart, use_container_width=True)
                
                # Display detailed ingredient needs
                st.subheader("Detailed Ingredient Requirements")
                
                # Get total needed for each ingredient for the forecast period
                total_ingredients = {}
                for date, ingredients in ingredient_forecast.items():
                    for ing, qty in ingredients.items():
                        total_ingredients[ing] = total_ingredients.get(ing, 0) + qty
                
                # Create a dataframe and display
                ingredients_summary = pd.DataFrame(
                    [(ing, qty) for ing, qty in total_ingredients.items()],
                    columns=['Ingredient', f'Total Needed (Next {forecast_days} Days)']
                ).sort_values(by=f'Total Needed (Next {forecast_days} Days)', ascending=False)
                
                st.dataframe(ingredients_summary)
        
        with tab4:
            st.header("Staffing Optimization")
            
            # Configuration for staffing model
            st.subheader("Staffing Model Configuration")
            
            col1, col2 = st.columns(2)
            
            with col1:
                orders_per_staff = st.number_input(
                    "Average orders handled per staff member per hour",
                    min_value=1,
                    max_value=20,
                    value=5
                )
                
                min_staff = st.number_input(
                    "Minimum staff required at all times",
                    min_value=1,
                    max_value=10,
                    value=2
                )
            
            with col2:
                prep_time_factor = st.slider(
                    "Preparation time factor",
                    min_value=0.5,
                    max_value=2.0,
                    value=1.0,
                    step=0.1,
                    help="Multiplier for estimated preparation time (higher means more staff needed)"
                )
                
                # Staff types
                staff_types = st.multiselect(
                    "Staff roles to consider",
                    options=["Chefs", "Waiters", "Kitchen helpers", "Bartenders"],
                    default=["Chefs", "Waiters"]
                )
            
            # Calculate staffing needs
            st.subheader("Staffing Requirements Forecast")
            with st.spinner("Calculating staffing needs..."):
                staffing_results = optimize_staffing(
                    forecast_result, 
                    orders_per_staff,
                    min_staff,
                    prep_time_factor,
                    staff_types
                )
                
                # Plot staffing requirements
                staff_chart = plot_staff_requirements(staffing_results)
                st.plotly_chart(staff_chart, use_container_width=True)
                
                # Display staffing recommendations table
                st.subheader("Daily Staffing Recommendations")
                
                staff_df = pd.DataFrame(staffing_results)
                staff_df['date'] = staff_df['date'].dt.strftime('%Y-%m-%d')
                staff_df = staff_df.set_index('date')
                
                st.dataframe(staff_df)
                
                # Staffing cost estimation
                st.subheader("Estimated Staffing Costs")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    hourly_rates = {}
                    for staff_type in staff_types:
                        hourly_rates[staff_type] = st.number_input(
                            f"Hourly rate for {staff_type} ($)",
                            min_value=5,
                            max_value=50,
                            value=15
                        )
                
                with col2:
                    shift_hours = st.number_input(
                        "Hours per shift",
                        min_value=4,
                        max_value=12,
                        value=8
                    )
                    
                    st.write("Total estimated staffing cost:")
                    
                    # Calculate total cost
                    total_cost = 0
                    for staff_type in staff_types:
                        staff_count = staff_df[staff_type].sum()
                        staff_cost = staff_count * hourly_rates[staff_type] * shift_hours
                        total_cost += staff_cost
                        st.write(f"- {staff_type}: ${staff_cost:.2f}")
                    
                    st.markdown(f"### Total: ${total_cost:.2f}")
    
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.exception(e)

# Add footer
st.markdown("---")
st.markdown("Hotel Order Analysis System | Powered by Streamlit & Prophet")
