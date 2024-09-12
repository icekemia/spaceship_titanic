from flask import Blueprint, Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import traceback

main = Blueprint("main", __name__)

ensemble_components = joblib.load("app/ensemble_components.pkl")
best_estimators = {key: joblib.load(f"app/{key}_model.pkl") for key in ensemble_components['best_estimators']}
transformer = joblib.load("app/transformer.pkl")

REQUIRED_FEATURES = [
    "PassengerId",
    "HomePlanet",
    "CryoSleep",
    "Cabin",
    "Destination",
    "Age",
    "VIP",
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "Name",
]

def preprocess_input(data):
    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Convert relevant fields to numeric types
    numeric_fields = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Age']
    for field in numeric_fields:
        df[field] = pd.to_numeric(df[field], errors='coerce')

    # Example feature engineering steps
    # Adjust these steps based on your actual EDA process

    # Create 'TotalExpenses' feature
    df['TotalExpenses'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

    # Create 'IsSpending' feature
    df['IsSpending'] = df['TotalExpenses'] > 0

    # Create 'GroupSize' and 'IsAlone' features
    df['GroupSize'] = df['PassengerId'].apply(lambda x: len(x.split('_')))
    df['IsAlone'] = df['GroupSize'] == 1

    # Create 'Deck' and 'Side' features from 'Cabin'
    df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notnull(x) else 'Unknown')
    df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[-1] if pd.notnull(x) else 'Unknown')

    # Create 'CabinOrdinal' feature
    df['CabinOrdinal'] = df['Cabin'].apply(lambda x: int(x.split('/')[1]) if pd.notnull(x) else -1)

    # Create 'ExpenseCategory' feature
    df['ExpenseCategory'] = pd.cut(df['TotalExpenses'], bins=[-1, 0, 1000, 5000, 10000, float('inf')],
                                   labels=['0', '0-1k', '1k-5k', '5k-10k', '10k+'])

    # Map 'ExpenseCategory' to numerical values
    expense_category_mapping = {
        "0": 0,
        "0-1k": 1,
        "1k-5k": 2,
        "5k-10k": 3,
        "10k+": 4,
    }
    df["ExpenseCategory"] = df["ExpenseCategory"].map(expense_category_mapping)

    # Replace boolean columns with integer values
    boolean_columns = ['CryoSleep', 'VIP']
    df[boolean_columns] = df[boolean_columns].replace({"True": True, "False": False})
    df[boolean_columns] = df[boolean_columns].astype(int)

    # Remove prefixes from column names if necessary
    def remove_prefixes(col):
        return col.split('_')[-1] if '_' in col else col

    df.columns = [remove_prefixes(col) for col in df.columns]

    # Apply log transformation to specified columns
    log_transform_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'TotalExpenses']
    for col in log_transform_columns:
        df[col] = np.log1p(df[col])

    # Drop unnecessary columns
    df.drop(["CabinNum", "GroupID"], axis=1, inplace=True, errors='ignore')

    return df

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
        input_data = preprocess_input(data)

        # Transform the input data
        input_transformed = transformer.transform(input_data)

        # Predict using the custom soft voting ensemble
        preds = np.zeros(input_transformed.shape[0])
        for key, classifier in best_estimators.items():
            preds += classifier.predict_proba(input_transformed)[:, 1]

        # Average the predictions
        preds = preds / len(best_estimators)

        # Convert the prediction to a native Python type
        prediction = int(preds[0] > 0.5)  # Assuming a threshold of 0.5 for binary classification

        return jsonify({"prediction": prediction})

    except Exception as e:
        # Print the traceback to help with debugging
        print("Error occurred: ", str(e))
        print(traceback.format_exc())

        # Return a 500 error with the exception message
        return jsonify({"error": "Internal server error", "message": str(e)}), 500
