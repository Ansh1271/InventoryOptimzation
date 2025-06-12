import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import datetime
import plotly.express as px
import plotly.graph_objects as go
import io
import os
import pickle

# Set page configuration
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Add title and description
st.title("Sales Forecasting Dashboard")
st.markdown("""
This dashboard allows you to visualize data and forecast future values using Random Forest regression.
Upload your dataset to generate forecasts.
""")

# Create a directory for saving models if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

MODEL_PATH = 'models/random_forest_model.pkl'

def load_uploaded_data(uploaded_file):
    try:
        data = pd.read_csv(uploaded_file)
        # Strip any extra spaces in the column names
        data.columns = data.columns.str.strip()
        return data
    except Exception as e:
        st.error(f"Error loading uploaded data: {e}")
        return None

def prepare_features(data, target_column):
    if data is None:
        return None
    
    # Create a copy to avoid modifying the original dataframe
    processed_data = data.copy()
    
    # Check if there's a date column
    date_col = None
    date_cols = []
    
    # Try to identify potential date columns
    for col in processed_data.columns:
        try:
            processed_data[f'{col}_temp'] = pd.to_datetime(processed_data[col], errors='coerce')
            if not processed_data[f'{col}_temp'].isna().all():
                date_cols.append(col)
            processed_data.drop(f'{col}_temp', axis=1, inplace=True)
        except:
            continue
    
    # Let user select the date column if multiple options
    if len(date_cols) > 0:
        if len(date_cols) == 1:
            date_col = date_cols[0]
        else:
            date_col = st.selectbox("Select the date column:", date_cols)
        
        # Convert selected date column to datetime
        processed_data['Date'] = pd.to_datetime(processed_data[date_col], errors='coerce')
        
        # Create time-based features if date column exists
        processed_data['day_of_week'] = processed_data['Date'].dt.day_of_week
        processed_data['month'] = processed_data['Date'].dt.month
        processed_data['quarter'] = processed_data['Date'].dt.quarter
        processed_data['day_of_year'] = processed_data['Date'].dt.day_of_year
        processed_data['week_of_year'] = processed_data['Date'].dt.isocalendar().week
        processed_data['is_weekend'] = processed_data['day_of_week'] >= 5
        
        # Add cyclic features for better seasonal representation
        processed_data['month_sin'] = np.sin(2 * np.pi * processed_data['month']/12)
        processed_data['month_cos'] = np.cos(2 * np.pi * processed_data['month']/12)
        processed_data['day_of_week_sin'] = np.sin(2 * np.pi * processed_data['day_of_week']/7)
        processed_data['day_of_week_cos'] = np.cos(2 * np.pi * processed_data['day_of_week']/7)
        
        # Create lag features for the target column
        lags = 3  # Reduced from 5 to be more flexible with smaller datasets
        for lag in range(1, lags + 1):
            processed_data[f'lag_{lag}'] = processed_data[target_column].shift(lag)
        
        # Add rolling statistics
        rolling_window = 3
        processed_data['rolling_mean'] = processed_data[target_column].rolling(window=rolling_window).mean()
        processed_data['rolling_std'] = processed_data[target_column].rolling(window=rolling_window).std()
    
    # Encode categorical columns
    cat_columns = processed_data.select_dtypes(include=['object', 'category']).columns
    for col in cat_columns:
        if col != date_col and col in processed_data.columns:  # Skip date column
            try:
                label_encoder = LabelEncoder()
                processed_data[col] = label_encoder.fit_transform(processed_data[col].fillna('MISSING'))
            except:
                # If encoding fails, drop the column
                processed_data.drop(col, axis=1, inplace=True)
    
    return processed_data

def get_features_list(data, target_column, date_column=None):
    # Get all columns except target and date
    if date_column:
        features = [col for col in data.columns if col != target_column and col != date_column and col != 'Date']
    else:
        features = [col for col in data.columns if col != target_column and col != 'Date']
    
    # Keep only numeric columns
    numeric_features = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(data[feature]):
            numeric_features.append(feature)
    
    return numeric_features

def train_model(data, target_column):
    # Get the list of numeric features
    features = get_features_list(data, target_column)
    
    # Ensure we have features to work with
    if not features:
        st.error("No usable numeric features found after data preparation.")
        return None, None, None, None, None, None
    
    # Remove rows with NaN values
    clean_data = data.dropna(subset=features + [target_column])
    
    # Define the features (X) and the target variable (y)
    X = clean_data[features]
    y = clean_data[target_column]
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Initialize and train the Random Forest model
    with st.spinner('Training Random Forest model...'):
        rf_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            n_jobs=-1,
            random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        # Save the model to a file
        with open(MODEL_PATH, 'wb') as file:
            pickle.dump(rf_model, file)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    
    return rf_model, X_train, X_test, y_train, y_test, rf_pred

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    return None

def evaluate_model(y_test, rf_pred):
    mse = mean_squared_error(y_test, rf_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, rf_pred)
    mae = mean_absolute_error(y_test, rf_pred)
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAE': mae
    }
    
    return metrics

def generate_forecast(data, rf_model, features, target_column, forecast_periods=6):
    """
    Generate forecasts using the trained model.
    For time series data, forecasts are made sequentially.
    For non-time series data, predictions are made for new inputs.
    """
    # Create forecast output based on data type
    if 'Date' in data.columns:
        # Time series forecast
        # Ensure data is sorted correctly
        data.sort_values(by='Date', inplace=True)
        
        # Get the last values for lag features
        last_values = list(data[target_column].tail(3))  # Get last 3 values (we use lag 1-3)
        
        # Generate future dates
        last_date = data['Date'].max()
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='M')
        
        predictions = []
        
        # Make predictions for each future period
        for i, date in enumerate(future_dates):
            future_row = pd.DataFrame(index=[0])
            
            # Add date-based features
            future_row['day_of_week'] = date.dayofweek
            future_row['month'] = date.month
            future_row['quarter'] = date.quarter
            future_row['day_of_year'] = date.dayofyear
            future_row['week_of_year'] = date.isocalendar().week
            future_row['is_weekend'] = date.dayofweek >= 5
            future_row['month_sin'] = np.sin(2 * np.pi * date.month/12)
            future_row['month_cos'] = np.cos(2 * np.pi * date.month/12)
            future_row['day_of_week_sin'] = np.sin(2 * np.pi * date.dayofweek/7)
            future_row['day_of_week_cos'] = np.cos(2 * np.pi * date.dayofweek/7)
            
            # Add lag features
            for j in range(min(3, len(last_values))):
                future_row[f'lag_{j+1}'] = last_values[-(j+1)]
            
            # Add rolling stats
            future_row['rolling_mean'] = np.mean(last_values)
            future_row['rolling_std'] = np.std(last_values) if len(last_values) > 1 else 0
            
            # Make prediction
            # Only include features that are in the model
            model_features = [f for f in features if f in future_row.columns]
            if len(model_features) < len(features):
                missing = set(features) - set(model_features)
                for f in missing:
                    future_row[f] = 0  # Use 0 as default for missing features
            
            prediction = rf_model.predict(future_row[features])[0]
            predictions.append(prediction)
            
            # Update last values for next iteration
            last_values.append(prediction)
            if len(last_values) > 3:
                last_values.pop(0)
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Value': predictions
        })
        
    else:
        # Non-time series forecast - just duplicate last few entries with predictions
        # Get last few rows for demonstration
        sample_rows = data.tail(forecast_periods).copy()
        
        # Make predictions on these sample rows
        predictions = rf_model.predict(sample_rows[features])
        
        # Create forecast dataframe
        forecast_df = sample_rows.copy()
        forecast_df['Predicted_Value'] = predictions
        forecast_df[target_column] = np.nan  # Clear actual values
    
    return forecast_df

# Main app function
def main():
    # Create sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Data Selection", "Data Overview", "Model Performance", "Forecast"])
    
    # Initialize session state if needed
    if 'data' not in st.session_state:
        st.session_state.data = None
        st.session_state.prepared_data = None
        st.session_state.rf_model = None
        st.session_state.trained = False
        st.session_state.target_column = None
        st.session_state.features = None

    if page == "Data Selection":
        st.header("Data Selection")
        st.write("Upload your dataset for forecasting using Random Forest regression.")
        
        uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
        
        if uploaded_file is not None:
            data = load_uploaded_data(uploaded_file)
            if data is not None:
                st.success("Dataset uploaded successfully!")
                st.write("Preview of the uploaded data:")
                st.dataframe(data.head())
                
                # Select target column
                numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    target_column = st.selectbox("Select target column (value to forecast):", numeric_cols)
                    st.session_state.target_column = target_column
                    st.session_state.data = data
                    
                    if st.button("Prepare Data and Train Model"):
                        st.session_state.prepared_data = prepare_features(data, target_column)
                        
                        if st.session_state.prepared_data is not None:
                            st.session_state.prepared_data = st.session_state.prepared_data.dropna()
                            
                            if len(st.session_state.prepared_data) > 5:  # Need at least 5 rows for training
                                st.session_state.rf_model, X_train, X_test, y_train, y_test, rf_pred = train_model(
                                    st.session_state.prepared_data, target_column)
                                
                                if st.session_state.rf_model is not None:
                                    st.session_state.trained = True
                                    st.session_state.features = get_features_list(
                                        st.session_state.prepared_data, target_column)
                                    st.success("Data prepared and model trained successfully!")
                            else:
                                st.error("Not enough data after preparation. Please check your dataset.")
                else:
                    st.error("No numeric columns found in the dataset. Please upload a dataset with numeric data.")
        
        # Load saved model if available and no model in session state
        if st.session_state.rf_model is None and os.path.exists(MODEL_PATH):
            if st.button("Load Saved Model"):
                st.session_state.rf_model = load_model()
                if st.session_state.rf_model is not None:
                    st.success("Saved model loaded successfully!")
                    st.session_state.trained = True
                else:
                    st.error("Failed to load saved model.")
    
    elif page == "Data Overview":
        if st.session_state.data is None:
            st.warning("Please upload data first from the 'Data Selection' page.")
            return
            
        st.header("Data Overview")
        
        # Show basic dataset information
        st.subheader("Dataset Information")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Total Records:** {len(st.session_state.data)}")
            if 'Date' in st.session_state.prepared_data.columns:
                st.write(f"**Date Range:** {st.session_state.prepared_data['Date'].min().date()} to {st.session_state.prepared_data['Date'].max().date()}")
        with col2:
            if st.session_state.target_column:
                st.write(f"**Target Column:** {st.session_state.target_column}")
                st.write(f"**Target Mean:** {st.session_state.data[st.session_state.target_column].mean():.2f}")
        
        # Show raw data sample
        st.subheader("Raw Data Sample")
        st.dataframe(st.session_state.data.head())
        
        # Show prepared data sample
        st.subheader("Prepared Data Sample")
        st.dataframe(st.session_state.prepared_data.head())
        
        # Show target column distribution
        if st.session_state.target_column:
            st.subheader(f"Distribution of {st.session_state.target_column}")
            fig = px.histogram(st.session_state.data, x=st.session_state.target_column)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show time series plot if date column exists
            if 'Date' in st.session_state.prepared_data.columns:
                st.subheader(f"Time Series: {st.session_state.target_column}")
                fig = px.line(
                    st.session_state.prepared_data.sort_values('Date'), 
                    x='Date', 
                    y=st.session_state.target_column, 
                    title=f'{st.session_state.target_column} Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Performance":
        if not st.session_state.trained:
            st.warning("Please train the model first on the 'Data Selection' page.")
            return
            
        if st.session_state.prepared_data is None:
            st.warning("No data has been prepared. Please go to 'Data Selection' page first.")
            return
            
        st.header("Model Performance")
        
        # Re-run the model evaluation for display
        model = st.session_state.rf_model
        data = st.session_state.prepared_data
        target = st.session_state.target_column
        features = st.session_state.features
        
        # Split data again for evaluation
        clean_data = data.dropna(subset=features + [target])
        X = clean_data[features]
        y = clean_data[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Make predictions and evaluate
        rf_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, rf_pred)
        
        # Display model metrics
        st.subheader("Random Forest Model Evaluation Metrics")
        metrics_df = pd.DataFrame({
            'MSE': [metrics['MSE']],
            'RMSE': [metrics['RMSE']],
            'R²': [metrics['R²']],
            'MAE': [metrics['MAE']]
        }, index=['Random Forest'])
        st.dataframe(metrics_df, use_container_width=True)
        
        # Show feature importance
        st.subheader("Feature Importance")
        feature_importances = model.feature_importances_
        feature_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False)
        
        fig = px.bar(feature_df, x='Importance', y='Feature', orientation='h',
                    title='Features by Importance', color='Importance', color_continuous_scale='viridis')
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show actual vs predicted
        st.subheader("Actual vs Predicted Values")
        
        # Create a dataframe with actual and predicted values
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': rf_pred
        })
        
        if 'Date' in data.columns:
            comparison_df['Date'] = data.loc[y_test.index, 'Date'].values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Actual'], 
                                    mode='lines+markers', name='Actual'))
            fig.add_trace(go.Scatter(x=comparison_df['Date'], y=comparison_df['Predicted'], 
                                    mode='lines+markers', name='Predicted'))
            fig.update_layout(title='Actual vs Predicted Values',
                            xaxis_title='Date',
                            yaxis_title=target)
        else:
            fig = px.scatter(comparison_df, x='Actual', y='Predicted', 
                           title='Actual vs Predicted Values')
            fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(),
                        line=dict(color="Red", width=2, dash="dash"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show error analysis
        st.subheader("Error Analysis")
        comparison_df['Error'] = comparison_df['Actual'] - comparison_df['Predicted']
        
        if 'Date' in data.columns:
            fig = px.line(comparison_df, x='Date', y='Error', title='Prediction Errors Over Time')
        else:
            fig = px.histogram(comparison_df, x='Error', title='Distribution of Prediction Errors')
        
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Forecast":
        if not st.session_state.trained:
            st.warning("Please train the model first on the 'Data Selection' page.")
            return
            
        if st.session_state.data is None:
            st.warning("No data available. Please go to 'Data Selection' page first.")
            return
        
        st.header("Forecast")
        
        # Allow user to select forecast period
        forecast_periods = st.slider("Select Forecast Periods", min_value=1, max_value=24, value=6)
        
        # Generate forecast
        forecast_df = generate_forecast(
            st.session_state.prepared_data, 
            st.session_state.rf_model,
            st.session_state.features,
            st.session_state.target_column,
            forecast_periods
        )
        
        # Display forecast table
        st.subheader("Forecasted Values")
        st.dataframe(forecast_df, use_container_width=True)
        
        # Display forecast chart
        st.subheader("Forecast Visualization")
        
        if 'Date' in forecast_df.columns:
            # For time series data
            # Combine historical and forecasted data for visualization
            historical = st.session_state.prepared_data[[
                'Date', st.session_state.target_column
            ]].copy()
            historical.columns = ['Date', 'Value']
            historical['Type'] = 'Historical'
            
            forecast = forecast_df[['Date', 'Predicted_Value']].copy()
            forecast.columns = ['Date', 'Value']
            forecast['Type'] = 'Forecast'
            
            combined = pd.concat([historical, forecast])
            
            fig = px.line(combined, x='Date', y='Value', color='Type', title='Historical Values and Forecast',
                        color_discrete_map={'Historical': 'blue', 'Forecast': 'red'})
            fig.update_layout(xaxis_title='Date', yaxis_title=st.session_state.target_column)
            
            # Add vertical line to separate historical and forecast
            fig.add_vline(x=historical['Date'].max(), line_dash="dash", line_color="gray")
        else:
            # For non-time series data
            fig = px.bar(forecast_df, y='Predicted_Value', title='Forecasted Values')
            fig.update_layout(yaxis_title=st.session_state.target_column)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add download button for forecast
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            label="Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    main()