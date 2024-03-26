import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved models
# Load the saved models
xgb_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/xgboost_model.joblib")
arima_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/arima_model.joblib")
rf_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/random_forest_model.joblib")
sarima_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/sarima_model.joblib")

# Load the dataset
data_path = r"C:\Users\wanji\Desktop\Sales tool 2\Sales-Tool-Analysis2\sales_data.csv"
sales_data = pd.read_csv(data_path)

# Streamlit App
st.title("Sales Forecasting")

# Sidebar
st.sidebar.header("Choose Prediction Model")

# Dropdown to select model
selected_model = st.sidebar.selectbox(
    "Select Model",
    ("XGBoost", "ARIMA", "Random Forest", "SARIMA")
)

# Display selected model
st.sidebar.subheader("Selected Model")
st.sidebar.write(selected_model)

# Main content
st.subheader("Sales Data")

# Display sample data
st.write(sales_data.head())

# Prediction function
# Prediction function
# Update the ARIMA and SARIMA prediction functions to include parameters
def predict_sales(model, data):
    # Get the number of features expected by the model
    num_features = None
    if model in ["XGBoost", "Random Forest"]:
        num_features = data.shape[0]  # Assuming the number of features is equal to the number of columns in the input data
    # Check if the number of features matches the model's expectations
    if num_features is not None and num_features != 37:  # Update the number of features accordingly
        st.error(f"Invalid number of features for {model} model!")
        return None
    # Make predictions based on the selected model
    if model == "XGBoost":
        prediction = xgb_model.predict(data.reshape(1, -1))
    elif model == "ARIMA":
        # Provide the appropriate parameters when calling predict() method
        prediction = arima_model.predict(n_periods=1, **arima_params)[0]  # Assuming you want to forecast one step ahead
    elif model == "Random Forest":
        prediction = rf_model.predict(data.reshape(1, -1))
    elif model == "SARIMA":
        # Provide the appropriate parameters when calling predict() method
        prediction = sarima_model.predict(n_periods=1, **sarima_params)[0]  # Assuming you want to forecast one step ahead
    return prediction

# Prediction
if st.button("Predict Sales"):
    # Prepare data for prediction
    input_data = sales_data.iloc[-1, 1:].values  # Exclude the first column (Date)

    # Make prediction
    predicted_sales = predict_sales(selected_model, input_data)

    # Display prediction
    if predicted_sales is not None:
        st.success(f"Predicted Sales: {predicted_sales}")  # Assuming prediction is a scalar value
