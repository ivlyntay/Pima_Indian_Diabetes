# Diabetes Prediction using Neural Networks

A machine learning project that predicts diabetes using a neural network model trained on the Pima Indians Diabetes dataset.

## 🚀 Features

- **Neural Network Model**: Multi-layer perceptron with TensorFlow/Keras
- **Data Preprocessing**: Handles missing values and feature scaling
- **Model Persistence**: Save and load trained models
- **Interactive Demo**: Make predictions for individual patients
- **Robust Error Handling**: Comprehensive error checking and validation
- **Easy Setup**: Simple installation with requirements.txt

## 📊 Dataset

The project uses the Pima Indians Diabetes dataset, which contains 768 samples with 8 features:

- **Pregnancies**: Number of times pregnant
- **Glucose**: Plasma glucose concentration
- **BloodPressure**: Diastolic blood pressure (mm Hg)
- **SkinThickness**: Triceps skin fold thickness (mm)
- **Insulin**: 2-Hour serum insulin (mu U/ml)
- **BMI**: Body mass index (weight in kg/(height in m)^2)
- **DiabetesPedigreeFunction**: Diabetes pedigree function
- **Age**: Age (years)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd diabetes-prediction-master
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.19+
- pandas 2.3+
- scikit-learn 1.7+
- numpy 2.1+
- joblib 1.5+

## 🎯 Usage

### 1. Train the Model

Run the training script to create and train the neural network:

```bash
python model.py
```

This will:
- Load and preprocess the dataset
- Train a neural network model
- Save the model to `model.json` and `model.weights.h5`
- Save the scaler to `scaler.save`
- Display training progress and final accuracy

### 2. Evaluate the Model

Test the trained model on validation data:

```bash
python predict.py
```

This will:
- Load the saved model and scaler
- Evaluate on validation data
- Display the validation accuracy

### 3. Interactive Demo

Use the demo script for individual predictions:

```bash
python demo_prediction.py
```

This provides:
- Example predictions for high and low risk patients
- Interactive mode to input custom patient data
- Risk assessment with probability scores

## 🏗️ Model Architecture

The neural network consists of:
- **Input Layer**: 8 features
- **Hidden Layer 1**: 12 neurons with ReLU activation
- **Hidden Layer 2**: 8 neurons with ReLU activation
- **Output Layer**: 1 neuron with sigmoid activation

**Training Configuration**:
- Optimizer: Adam
- Loss Function: Binary Crossentropy
- Metrics: Accuracy
- Epochs: 100
- Batch Size: 10

## 📈 Performance

The model achieves approximately **79-80% accuracy** on the test set.

## 📁 Project Structure

```
diabetes-prediction-master/
├── model.py                 # Main training script
├── predict.py              # Model evaluation script
├── demo_prediction.py      # Interactive demo
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── pima-indians-diabetes.data  # Dataset
├── model.json             # Saved model architecture (generated)
├── model.weights.h5       # Saved model weights (generated)
└── scaler.save           # Saved feature scaler (generated)
```

## 🔧 Data Preprocessing

The preprocessing pipeline includes:

1. **Missing Value Handling**: Replace invalid zeros with NaN for specific features
2. **Imputation**: Fill missing values with median values
3. **Feature Scaling**: MinMax normalization (0-1 range)
4. **Train-Test Split**: 80% training, 20% testing

## 🎨 Example Usage

```python
from demo_prediction import predict_diabetes

# Example prediction
prediction, probability = predict_diabetes(
    pregnancies=6,
    glucose=148,
    blood_pressure=72,
    skin_thickness=35,
    insulin=0,
    bmi=33.6,
    diabetes_pedigree=0.627,
    age=50
)

print(f"Prediction: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
print(f"Probability: {probability:.3f}")
```

## ⚠️ Important Notes

- Ensure all required files are present before running predictions
- The model is for educational purposes and should not replace medical diagnosis
- Missing values in input data are handled automatically
- All features should be provided for accurate predictions

## 🐛 Troubleshooting

**Common Issues**:

1. **Import Errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **File Not Found**: Run `model.py` first to generate required files

3. **Version Conflicts**: Use the specified versions in requirements.txt

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Dataset: Pima Indians Diabetes Database from UCI Machine Learning Repository
- Built with TensorFlow/Keras and scikit-learn
- Inspired by machine learning best practices for healthcare applications
