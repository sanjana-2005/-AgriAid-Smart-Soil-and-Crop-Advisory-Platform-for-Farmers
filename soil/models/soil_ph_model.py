import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def rgb_to_hsv(r, g, b):
    """Convert RGB to HSV color space using OpenCV."""
    color = np.uint8([[[r, g, b]]])
    hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv[0][0]

def load_and_preprocess_data(file_path):
    """Load, convert RGB to HSV, and preprocess the soil pH dataset."""
    df = pd.read_csv(file_path)

    # Convert RGB to HSV
    df[['H', 'S', 'V']] = df.apply(lambda row: pd.Series(rgb_to_hsv(row['R'], row['G'], row['B'])), axis=1)

    # Features (HSV) and target (pH)
    X = df[['H', 'S', 'V']].values
    y = df['pH'].values

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def train_model(X_train, y_train, model_type='random_forest'):
    """Train a model (Random Forest or Linear Regression)."""
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError("Invalid model_type. Choose 'random_forest' or 'linear'")
    
    model.fit(X_train, y_train)
    return model

def evaluate_model_cv(model, X, y):
    """Evaluate the model using cross-validation."""
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Cross-Validated R² Scores: {scores}")
    print(f"Average R² Score: {scores.mean():.4f}")
    return scores

def evaluate_model_test(model, X_test, y_test):
    """Evaluate the model on a test set."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Test Set Mean Squared Error: {mse:.4f}")
    print(f"Test Set R² Score: {r2:.4f}")
    return y_pred

def plot_results(y_test, y_pred):
    """Plot actual vs predicted pH values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual pH")
    plt.ylabel("Predicted pH")
    plt.title("Actual vs Predicted Soil pH")
    plt.tight_layout()
    plt.savefig("soil_ph_prediction_results.png")
    plt.close()

def plot_feature_importance(model, feature_names=['H', 'S', 'V']):
    """Plot feature importance (for tree-based models)."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        plt.figure(figsize=(8, 4))
        sns.barplot(x=feature_names, y=importance)
        plt.title("Feature Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        plt.close()

def save_model(model, scaler, model_path="soil_ph_model.joblib", scaler_path="soil_ph_scaler.joblib"):
    """Save the trained model and scaler."""
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model to: {model_path}")
    print(f"Saved scaler to: {scaler_path}")

def predict_ph(model, scaler, rgb_values):
    """Predict soil pH from raw RGB values."""
    hsv_values = rgb_to_hsv(*rgb_values)
    hsv_scaled = scaler.transform(np.array(hsv_values).reshape(1, -1))
    ph_prediction = model.predict(hsv_scaled)[0]
    return ph_prediction

def main():
    file_path = r"C:\Users\shalu\Downloads\soil\soilpH_rgb.csv"
    model_type = 'random_forest'  # Change to 'linear' if you want Linear Regression

    print("🔄 Step 1: Loading and preprocessing data...")
    X_scaled, y, scaler = load_and_preprocess_data(file_path)

    print("\n🔄 Step 2: Training model...")
    model = train_model(X_scaled, y, model_type=model_type)

    print("\n🔍 Step 3: Evaluating model with cross-validation...")
    evaluate_model_cv(model, X_scaled, y)

    # For visualization, also use train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = evaluate_model_test(model, X_test, y_test)

    print("\n📊 Step 4: Generating plots...")
    plot_results(y_test, y_pred)
    plot_feature_importance(model)

    print("\n💾 Step 5: Saving model and scaler...")
    save_model(model, scaler)

    # Example prediction
    example_rgb = [150, 100, 50]
    ph = predict_ph(model, scaler, example_rgb)
    print(f"\n🧪 Example prediction for RGB {example_rgb}: Estimated pH = {ph:.2f}")

    print("\n✅ All steps completed successfully!")

if __name__ == "__main__":
    main()
