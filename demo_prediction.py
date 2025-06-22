"""
Demo script to make individual diabetes predictions using the trained model.
"""

from tensorflow.keras.models import model_from_json
import numpy as np
import joblib
import os

def load_model():
    """Load the trained model and scaler."""
    try:
        # Check if required files exist
        required_files = ["model.json", "model.weights.h5", "scaler.save"]
        for file in required_files:
            if not os.path.exists(file):
                raise FileNotFoundError(f"Required file '{file}' not found. Please run model.py first to train the model.")
        
        # Load model
        with open("model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights("model.weights.h5")
        
        # Load scaler
        scaler = joblib.load("scaler.save")
        
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness, 
                    insulin, bmi, diabetes_pedigree, age):
    """
    Predict diabetes for a single patient.
    
    Parameters:
    - pregnancies: Number of times pregnant
    - glucose: Plasma glucose concentration
    - blood_pressure: Diastolic blood pressure (mm Hg)
    - skin_thickness: Triceps skin fold thickness (mm)
    - insulin: 2-Hour serum insulin (mu U/ml)
    - bmi: Body mass index (weight in kg/(height in m)^2)
    - diabetes_pedigree: Diabetes pedigree function
    - age: Age (years)
    
    Returns:
    - prediction: 0 (no diabetes) or 1 (diabetes)
    - probability: Probability of having diabetes
    """
    
    # Load model and scaler
    model, scaler = load_model()
    if model is None or scaler is None:
        return None, None
    
    # Prepare input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                           insulin, bmi, diabetes_pedigree, age]])
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    probability = model.predict(input_scaled)[0][0]
    prediction = 1 if probability > 0.5 else 0
    
    return prediction, probability

def main():
    """Demo function with example predictions."""
    print("Diabetes Prediction Demo")
    print("=" * 30)
    
    # Example 1: High risk patient
    print("\nExample 1: High risk patient")
    pred, prob = predict_diabetes(
        pregnancies=6, glucose=148, blood_pressure=72, skin_thickness=35,
        insulin=0, bmi=33.6, diabetes_pedigree=0.627, age=50
    )
    if pred is not None:
        print(f"Prediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
        print(f"Probability: {prob:.3f}")
    
    # Example 2: Low risk patient
    print("\nExample 2: Low risk patient")
    pred, prob = predict_diabetes(
        pregnancies=1, glucose=85, blood_pressure=66, skin_thickness=29,
        insulin=0, bmi=26.6, diabetes_pedigree=0.351, age=31
    )
    if pred is not None:
        print(f"Prediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
        print(f"Probability: {prob:.3f}")
    
    # Interactive prediction
    print("\n" + "=" * 30)
    print("Interactive Prediction")
    print("Enter patient data for prediction:")
    
    try:
        pregnancies = int(input("Number of pregnancies: "))
        glucose = float(input("Glucose level: "))
        blood_pressure = float(input("Blood pressure: "))
        skin_thickness = float(input("Skin thickness: "))
        insulin = float(input("Insulin level: "))
        bmi = float(input("BMI: "))
        diabetes_pedigree = float(input("Diabetes pedigree function: "))
        age = int(input("Age: "))
        
        pred, prob = predict_diabetes(pregnancies, glucose, blood_pressure, skin_thickness,
                                    insulin, bmi, diabetes_pedigree, age)
        
        if pred is not None:
            print(f"\nPrediction: {'Diabetes' if pred == 1 else 'No Diabetes'}")
            print(f"Probability: {prob:.3f}")
            
            if prob > 0.7:
                print("⚠️  High risk - Consider consulting a healthcare professional")
            elif prob > 0.3:
                print("⚠️  Moderate risk - Monitor health indicators")
            else:
                print("✅ Low risk - Maintain healthy lifestyle")
        
    except ValueError:
        print("Invalid input. Please enter numeric values.")
    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

if __name__ == "__main__":
    main()
