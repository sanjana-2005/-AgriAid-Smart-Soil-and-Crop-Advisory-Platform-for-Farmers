# Soil pH Prediction Model

This model predicts soil pH values based on RGB color values extracted from soil images.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure your dataset (`soilpH_rgb.csv`) is in the correct format:
   - Columns: 'R', 'G', 'B' (RGB values), 'pH' (target variable)
   - Values: R, G, B should be integers 0-255, pH should be float

## Training the Model

Run the training script:
```bash
python soil_ph_model.py
```

This will:
1. Load and preprocess the data
2. Train a Random Forest model
3. Generate performance metrics
4. Create visualization plots
5. Save the trained model and scaler

## Output Files

- `soil_ph_model.joblib`: Trained model
- `soil_ph_scaler.joblib`: Feature scaler
- `soil_ph_prediction_results.png`: Actual vs Predicted plot
- `feature_importance.png`: Feature importance visualization

## Using the Model

To use the trained model for predictions:

```python
import joblib

# Load the model and scaler
model = joblib.load('soil_ph_model.joblib')
scaler = joblib.load('soil_ph_scaler.joblib')

# Make a prediction
rgb_values = [150, 100, 50]  # Example RGB values
rgb_array = np.array(rgb_values).reshape(1, -1)
rgb_scaled = scaler.transform(rgb_array)
ph_prediction = model.predict(rgb_scaled)[0]
```

## Model Performance

The model's performance metrics include:
- Mean Squared Error (MSE)
- R² Score

These metrics are printed during training and can be used to assess the model's accuracy. 
