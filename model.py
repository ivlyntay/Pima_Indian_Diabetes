import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import model_from_json
import joblib

def main():
    """Main function to train the diabetes prediction model."""
    try:
        # Check if data file exists
        data_file = "pima-indians-diabetes.data"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file '{data_file}' not found. Please ensure the file exists in the current directory.")

        print("Loading dataset...")
        # Load dataset
        df = pd.read_csv(data_file, header=None)
        df.columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                      'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

        print(f"Dataset loaded successfully. Shape: {df.shape}")

        # Replace invalid 0 values with NaN for specific columns
        print("Preprocessing data...")
        cols_with_zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[cols_with_zero_as_missing] = df[cols_with_zero_as_missing].replace(0, np.nan)

        # Impute missing values with median
        df.fillna(df.median(), inplace=True)

        # Split features and target
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Normalize features
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # Split into train/test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=7)
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        # Define MLP model
        print("Building neural network model...")
        model = Sequential()
        model.add(Dense(12, input_dim=8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Model compiled successfully.")

        # Train the model
        print("Training model...")
        model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=1)

        # Evaluate the model
        print("Evaluating model...")
        scores = model.evaluate(X_test, y_test)
        print("Model Accuracy: %.2f%%" % (scores[1]*100))

        # Save model to JSON
        print("Saving model...")
        model_json = model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)

        # Save weights
        model.save_weights("model.weights.h5")
        print("Model saved to model.json and model.weights.h5")

        # Save scaler for later use
        joblib.dump(scaler, "scaler.save")
        print("Scaler saved to scaler.save")

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
        print("Model training completed successfully!")
    else:
        print("Model training failed!")
