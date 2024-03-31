import streamlit as st
import pandas as pd
import pickle
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load the pickled models
with open("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/arima_model.pkl", "rb") as f:
    arima_model = pickle.load(f)

with open("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/sarima_model.pkl", "rb") as f:
    sarima_model = pickle.load(f)

# Streamlit App
st.title("Sales Forecasting")

# Sidebar
st.sidebar.header("Choose Prediction Model")

# Dropdown to select model
selected_model = st.sidebar.selectbox(
    "Select Model",
    ("ARIMA", "SARIMA")
)

# Display selected model
st.sidebar.subheader("Selected Model")
st.sidebar.write(selected_model)

# Main content
st.subheader("Sales Data")

# Load the dataset
data_path = r"C:\Users\wanji\Desktop\Sales tool 2\Sales-Tool-Analysis2\sales_data.csv"
# Load the dataset with appropriate date parsing and column specification
sales_data = pd.read_csv(data_path, parse_dates=['Period'])

# Set 'Period' column as index
sales_data.set_index('Period', inplace=True)

# Splitting the data into training and testing sets based on specific dates
cutoff_date = pd.Timestamp('2017-01-01')
train = sales_data[sales_data.index < cutoff_date]
test = sales_data[sales_data.index >= cutoff_date]
##
# Display sample data
st.write(sales_data.head())

def predict_sales(model, model_params, test_data):
    # Make predictions based on the selected model
    if model == "ARIMA":
        if not test_data.empty:  # Check if test_data is not empty
            # Use get_forecast method for ARIMA models
            forecast = model_params['model'].get_forecast(steps=len(test_data))
            prediction = forecast.predicted_mean
        else:
            prediction = None  # Return None if test_data is empty
    elif model == "SARIMA":
        if not test_data.empty:  # Check if test_data is not empty
            # Use get_prediction method for SARIMA models
            forecast = model_params['model'].get_prediction(start=test_data.index[0], end=test_data.index[-1], dynamic=False)
            prediction = forecast.predicted_mean
        else:
            prediction = None  # Return None if test_data is empty
    else:
        raise ValueError("Invalid model selected")
    return prediction
#

# PREDICTION
if st.button("Predict Sales"):
    if test.empty:  # Check if test data is empty before making predictions
        st.warning("Test data is empty. Cannot make prediction.")
    else:
        # Make prediction
        if selected_model == "ARIMA":
            predicted_sales = predict_sales("ARIMA", {'model': arima_model}, test)
        elif selected_model == "SARIMA":
            predicted_sales = predict_sales("SARIMA", {'model': sarima_model}, test)
        else:
            raise ValueError("Invalid model selected")

        # Display predicted sales
        if predicted_sales is not None:
            st.success(f"Predicted sales: {predicted_sales}")
        else:
            st.error("Failed to make prediction. Please check your data.")
