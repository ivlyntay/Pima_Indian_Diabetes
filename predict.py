from tensorflow.keras.models import model_from_json
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
import joblib

def main():
    """Main function to load model and make predictions."""
    try:
        # Check if required files exist
        required_files = ["model.json", "model.weights.h5", "scaler.save", "pima-indians-diabetes.data"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file '{file}' not found. Please run model.py first to train the model.")

        print("Loading model...")
        # Load model
        with open("model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)

        # Load weights
        loaded_model.load_weights("model.weights.h5")
        print("Model loaded successfully from disk")

        print("Loading and preprocessing data...")
        # Load dataset
        df = pd.read_csv("pima-indians-diabetes.data", header=None)
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                      'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

        # Replace invalid 0 values with NaN for specific columns
        cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

        # Impute with median
        df.fillna(df.median(), inplace=True)

        # Split features and target
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Load scaler and apply
        print("Loading scaler and transforming data...")
        scaler = joblib.load("scaler.save")
        X_scaled = scaler.transform(X)

        # Split into validation set
        _, X_validation, _, y_validation = train_test_split(X_scaled, y, test_size=0.2, random_state=7)

        # Compile model
        loaded_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        # Evaluate
        print("Evaluating model on validation data...")
        scores = loaded_model.evaluate(X_validation, y_validation, verbose=0)
        print("Validation Accuracy: %.2f%%" % (scores[1] * 100))

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("Prediction completed successfully!")
    else:
        print("Prediction failed!")
