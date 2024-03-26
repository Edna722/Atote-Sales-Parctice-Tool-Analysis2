from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load joblib models
arima_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/arima_model.joblib")
random_forest_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/random_forest_model.joblib")
sarima_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/sarima_model.joblib")
xgboost_model = joblib.load("C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/xgboost_model.joblib")

# Dataset paths
training_data_path = "C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/Datasets/TrainingData.csv"
test_data_path = "C:/Users/wanji/Desktop/Sales tool 2/Sales-Tool-Analysis2/Datasets/TestData.csv"

# Define routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            return "No file uploaded"

        # Read the uploaded CSV file
        data = pd.read_csv(file)

        # Make predictions using the loaded models
        arima_prediction = arima_model.predict(data)
        random_forest_prediction = random_forest_model.predict(data)
        sarima_prediction = sarima_model.predict(data)
        xgboost_prediction = xgboost_model.predict(data)

        # Render the prediction results page
        return render_template("results.html", arima_prediction=arima_prediction,
                               random_forest_prediction=random_forest_prediction,
                               sarima_prediction=sarima_prediction,
                               xgboost_prediction=xgboost_prediction)

if __name__ == "__main__":
    app.run(debug=True)
