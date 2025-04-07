import joblib
import numpy as np
from pathlib import Path
import cv2

class SoilHealthAnalyzer:
    def __init__(self, model_path, scaler_path):
        """Initialize the soil health analyzer with pH prediction model."""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def _predict_ph(self, rgb_values):
        """Predict soil pH from RGB values using the loaded model."""
        color = np.uint8([[[rgb_values[0], rgb_values[1], rgb_values[2]]]])
        hsv = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)[0][0]
        hsv_scaled = self.scaler.transform(np.array(hsv).reshape(1, -1))
        return self.model.predict(hsv_scaled)[0]
    
    def _evaluate_ph_health(self, ph):
        """Evaluate soil health based on pH level."""
        if ph < 4.5:
            return "Poor", "Extremely acidic soil. Most crops cannot grow well."
        elif 4.5 <= ph < 5.5:
            return "Fair", "Strongly acidic soil. Limited crop options."
        elif 5.5 <= ph < 6.5:
            return "Good", "Slightly acidic soil. Ideal for most crops."
        elif 6.5 <= ph < 7.5:
            return "Excellent", "Neutral soil. Optimal for most plants."
        elif 7.5 <= ph < 8.5:
            return "Good", "Slightly alkaline soil. Good for specific crops."
        elif 8.5 <= ph < 9.0:
            return "Fair", "Strongly alkaline soil. Limited crop options."
        else:
            return "Poor", "Extremely alkaline soil. Most crops cannot grow well."
    
    def _get_crop_recommendations(self, ph):
        """Get crop recommendations based on soil pH."""
        if ph < 5.0:
            return ["Blueberries", "Potatoes", "Sweet Potatoes"]
        elif 5.0 <= ph < 6.0:
            return ["Strawberries", "Tomatoes", "Carrots"]
        elif 6.0 <= ph < 7.0:
            return ["Corn", "Wheat", "Soybeans", "Most vegetables"]
        elif 7.0 <= ph < 8.0:
            return ["Asparagus", "Beets", "Cabbage"]
        else:
            return ["Date Palms", "Figs", "Some Bean varieties"]
    
    def _get_improvement_recommendations(self, ph):
        """Get recommendations for improving soil health based on pH."""
        if ph < 6.0:
            return [
                "Add agricultural lime to raise pH",
                "Apply organic matter like composted leaves",
                "Consider wood ash application",
                "Use dolomitic limestone for magnesium deficiency"
            ]
        elif ph > 7.5:
            return [
                "Add sulfur to lower pH",
                "Use acidifying fertilizers",
                "Apply organic matter like pine needles",
                "Consider aluminum sulfate for quick pH reduction"
            ]
        else:
            return [
                "Maintain current soil conditions",
                "Add organic matter regularly",
                "Use balanced fertilizers",
                "Practice crop rotation"
            ]
    
    def analyze_soil_health(self, rgb_values):
        """
        Analyze soil health based on RGB values from soil image.
        Returns a comprehensive soil health report.
        """
        try:
            # Predict pH
            ph = self._predict_ph(rgb_values)
            
            # Evaluate soil health
            health_status, health_description = self._evaluate_ph_health(ph)
            
            # Get recommendations
            crop_recommendations = self._get_crop_recommendations(ph)
            improvement_recommendations = self._get_improvement_recommendations(ph)
            
            # Create soil health report
            report = {
                "soil_ph": round(ph, 2),
                "health_status": health_status,
                "health_description": health_description,
                "suitable_crops": crop_recommendations,
                "improvement_recommendations": improvement_recommendations
            }
            
            return report
            
        except Exception as e:
            return {
                "error": f"Error analyzing soil health: {str(e)}",
                "soil_ph": None,
                "health_status": "Unknown",
                "health_description": "Could not determine soil health",
                "suitable_crops": [],
                "improvement_recommendations": ["Please consult a soil expert for detailed analysis"]
            }

def main():
    """Example usage of the SoilHealthAnalyzer."""
    # Model paths
    model_path = r"C:\Users\shalu\Downloads\soil\models\soil_ph_model.joblib"
    scaler_path = r"C:\Users\shalu\Downloads\soil\models\soil_ph_scaler.joblib"
    
    # Initialize analyzer
    analyzer = SoilHealthAnalyzer(model_path, scaler_path)
    
    # Example RGB values
    example_rgb = [150, 100, 50]  # Example soil color
    
    # Get soil health analysis
    print("\n🔬 Analyzing soil health...")
    report = analyzer.analyze_soil_health(example_rgb)
    
    # Print report
    print("\n📋 Soil Health Report")
    print("=" * 50)
    print(f"Soil pH: {report['soil_ph']}")
    print(f"Health Status: {report['health_status']}")
    print(f"Description: {report['health_description']}")
    print("\n🌱 Suitable Crops:")
    for crop in report['suitable_crops']:
        print(f"- {crop}")
    print("\n💡 Improvement Recommendations:")
    for rec in report['improvement_recommendations']:
        print(f"- {rec}")

if __name__ == "__main__":
    main() 