from flask import Blueprint, Flask, request, jsonify, render_template
import joblib
import pandas as pd
import traceback

main = Blueprint("main", __name__)

model = joblib.load("app/ensemble_components.pkl")

REQUIRED_FEATURES = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin,Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name",
]


@main.route("/")
def home():
    return render_template("index.html")


@main.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        # Identify missing and extra features
        missing_features = [
            feature for feature in REQUIRED_FEATURES if feature not in data
        ]
        extra_features = [
            feature for feature in data if feature not in REQUIRED_FEATURES
        ]

        if missing_features or extra_features:
            return (
                jsonify(
                    {
                        "error": "Invalid input",
                        "missing_features": missing_features,
                        "extra_features": extra_features,
                    }
                ),
                400,
            )

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Convert the prediction to a native Python type
        prediction = int(prediction)

        return jsonify({"prediction": prediction})

    except Exception as e:
        # Print the traceback to help with debugging
        print("Error occurred: ", str(e))
        print(traceback.format_exc())

        # Return a 500 error with the exception message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500
