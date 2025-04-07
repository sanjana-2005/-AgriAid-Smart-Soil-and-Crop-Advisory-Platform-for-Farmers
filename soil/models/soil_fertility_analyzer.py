class SoilFertilityAnalyzer:
    def __init__(self):
        # Define threshold values for nutrients
        self.thresholds = {
            'nitrogen': {'low': 100, 'medium': 200},
            'phosphorus': {'low': 20, 'medium': 40},
            'potassium': {'low': 150, 'medium': 300},
            'ph': {'low': 6.0, 'medium': 7.0},
            'organic_matter': {'low': 2, 'medium': 4}
        }

    def analyze_fertility(self, soil_data):
        """
        Analyze soil fertility based on input parameters
        """
        # Initialize score
        score = 0
        status_class = ""
        recommendations = []
        
        # Analyze each parameter
        nitrogen_status = self._analyze_parameter('nitrogen', soil_data['nitrogen'])
        phosphorus_status = self._analyze_parameter('phosphorus', soil_data['phosphorus'])
        potassium_status = self._analyze_parameter('potassium', soil_data['potassium'])
        ph_status = self._analyze_parameter('ph', soil_data['ph'])
        organic_matter_status = self._analyze_parameter('organic_matter', soil_data['organic_matter'])
        
        # Calculate overall score (0-10)
        parameter_scores = {
            'low': 1,
            'medium': 2,
            'high': 3
        }
        
        total_score = sum([
            parameter_scores[nitrogen_status],
            parameter_scores[phosphorus_status],
            parameter_scores[potassium_status],
            parameter_scores[ph_status],
            parameter_scores[organic_matter_status]
        ])
        
        # Convert to 10-point scale
        score = (total_score / 15) * 10
        
        # Determine overall status
        if score >= 8:
            status = "Excellent"
            status_class = "success"
        elif score >= 6:
            status = "Good"
            status_class = "info"
        elif score >= 4:
            status = "Fair"
            status_class = "warning"
        else:
            status = "Poor"
            status_class = "danger"
            
        # Generate recommendations
        recommendations = self._generate_recommendations(
            nitrogen_status,
            phosphorus_status,
            potassium_status,
            ph_status,
            organic_matter_status
        )
        
        return {
            'score': round(score, 1),
            'status': status,
            'status_class': status_class,
            'nitrogen_status': self._get_bootstrap_class(nitrogen_status),
            'phosphorus_status': self._get_bootstrap_class(phosphorus_status),
            'potassium_status': self._get_bootstrap_class(potassium_status),
            'nitrogen_level': nitrogen_status.title(),
            'phosphorus_level': phosphorus_status.title(),
            'potassium_level': potassium_status.title(),
            'recommendations': recommendations
        }

    def _analyze_parameter(self, parameter, value):
        """
        Analyze individual parameter values
        """
        thresholds = self.thresholds[parameter]
        if value < thresholds['low']:
            return 'low'
        elif value < thresholds['medium']:
            return 'medium'
        else:
            return 'high'

    def _get_bootstrap_class(self, status):
        """
        Convert status to Bootstrap color class
        """
        status_classes = {
            'low': 'danger',
            'medium': 'warning',
            'high': 'success'
        }
        return status_classes.get(status, 'secondary')

    def _generate_recommendations(self, n_status, p_status, k_status, ph_status, om_status):
        """
        Generate specific recommendations based on soil analysis
        """
        recommendations = []
        
        # Nitrogen recommendations
        if n_status == 'low':
            recommendations.append("Apply nitrogen-rich fertilizers like urea or ammonium sulfate")
        elif n_status == 'medium':
            recommendations.append("Maintain nitrogen levels with balanced fertilization")
            
        # Phosphorus recommendations
        if p_status == 'low':
            recommendations.append("Add phosphate fertilizers or bone meal to improve phosphorus content")
        elif p_status == 'medium':
            recommendations.append("Monitor phosphorus levels and maintain with regular fertilization")
            
        # Potassium recommendations
        if k_status == 'low':
            recommendations.append("Apply potassium-rich fertilizers like potassium chloride")
        elif k_status == 'medium':
            recommendations.append("Maintain potassium levels with balanced fertilization")
            
        # pH recommendations
        if ph_status == 'low':
            recommendations.append("Add lime to increase soil pH")
        elif ph_status == 'high':
            recommendations.append("Add sulfur to decrease soil pH")
            
        # Organic matter recommendations
        if om_status == 'low':
            recommendations.append("Add compost or organic matter to improve soil structure")
        elif om_status == 'medium':
            recommendations.append("Continue adding organic matter through crop residues or compost")
            
        return recommendations 