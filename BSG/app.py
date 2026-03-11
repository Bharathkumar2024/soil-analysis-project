from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load models
soil_model = joblib.load("models/soil_model.pkl")
crop_model = joblib.load("models/crop_model.pkl")
irrigation_model = joblib.load("models/irrigation_model.pkl")
fertility_model = joblib.load("models/fertility_model.pkl")


# Homepage
@app.route("/")
def home():
    return render_template("index.html")


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    features = np.array([[
        data["pH"],
        data["Nitrogen"],
        data["Phosphorus"],
        data["Potassium"],
        data["Moisture"],
        data["Organic_Carbon"]
    ]])

    soil = soil_model.predict(features)[0]
    crop = crop_model.predict(features)[0]
    irrigation = irrigation_model.predict(features)[0]
    fertility = fertility_model.predict(features)[0]

    return jsonify({
        "Soil_Type": soil,
        "Crop_Category": crop,
        "Irrigation_Type": irrigation,
        "Fertility_Score": round(float(fertility), 2)
    })


if __name__ == "__main__":
    app.run()