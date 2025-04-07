from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_from_directory
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from datetime import datetime
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.api_core import retry
import traceback
from flask_cors import CORS
from werkzeug.utils import secure_filename
from models.soil_health_analyzer import SoilHealthAnalyzer
from PIL import Image
import cv2

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

# Load models
SOIL_MODEL_PATH = os.path.join('models', 'soil_classification_model_03.h5')
PLANT_DISEASE_MODEL_PATH = r"C:\Users\shalu\Downloads\soil\models\plant_disease_cnn.keras"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

try:
    soil_model = tf.keras.models.load_model(SOIL_MODEL_PATH)
    plant_disease_model = tf.keras.models.load_model(PLANT_DISEASE_MODEL_PATH)
    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    soil_model = None
    plant_disease_model = None

# Soil classification labels
SOIL_LABELS = ['Alluvial Soil', 'Black Soil', 'Clay Soil', 'Red Soil']

# Plant disease labels
PLANT_DISEASE_LABELS = ["curl", "healthy", "slug", "spot"]

# Crop recommendations based on soil type
CROP_RECOMMENDATIONS = {
    'Alluvial Soil': ['Rice', 'Wheat', 'Sugarcane', 'Cotton', 'Jute'],
    'Black Soil': ['Cotton', 'Sugarcane', 'Jowar', 'Wheat', 'Linseed'],
    'Clay Soil': ['Rice', 'Wheat', 'Barley', 'Oats', 'Potatoes'],
    'Red Soil': ['Groundnut', 'Pulses', 'Millets', 'Tobacco', 'Fruits']
}

# Cultivation practices
CULTIVATION_PRACTICES = {
    'Alluvial Soil': {
        'General Practices': [
            'Maintain proper drainage to prevent waterlogging',
            'Regular soil testing for nutrient management',
            'Add organic matter to improve soil structure',
            'Practice crop rotation to maintain soil fertility',
            'Optimal pH range: 6.5-7.5'
        ],
        'Rice': [
            'Land preparation: Puddling and leveling',
            'Seed rate: 20-25 kg/ha for transplanting',
            'Spacing: 20x10 cm or 20x15 cm',
            'Water management: Maintain 5-7 cm water level',
            'Fertilizer: N:P:K - 120:60:60 kg/ha',
            'Harvesting: When 80% grains turn golden yellow'
        ],
        'Wheat': [
            'Sowing time: October-November',
            'Seed rate: 100-125 kg/ha',
            'Row spacing: 20-22.5 cm',
            'Irrigation: 4-6 times during crop growth',
            'Fertilizer: N:P:K - 120:60:40 kg/ha',
            'Harvesting: When grains are hard and golden'
        ],
        'Sugarcane': [
            'Planting time: February-March',
            'Seed rate: 40,000-45,000 setts/ha',
            'Row spacing: 90-100 cm',
            'Regular irrigation at 7-10 days interval',
            'Fertilizer: N:P:K - 250:100:125 kg/ha',
            'Harvesting: When brix reading is above 18%'
        ],
        'Cotton': [
            'Sowing time: April-May',
            'Seed rate: 15-20 kg/ha',
            'Spacing: 90x60 cm',
            'Irrigation: At critical growth stages',
            'Fertilizer: N:P:K - 120:60:60 kg/ha',
            'Pest management: Regular monitoring essential'
        ],
        'Jute': [
            'Sowing time: March-April',
            'Seed rate: 7-8 kg/ha',
            'Row spacing: 30 cm',
            'Regular weeding in initial stages',
            'Fertilizer: N:P:K - 80:40:40 kg/ha',
            'Harvesting: At yellow ripening stage'
        ]
    },
    'Black Soil': {
        'General Practices': [
            'Deep ploughing during summer',
            'Proper drainage system essential',
            'Mulching to prevent water loss',
            'Avoid over-irrigation',
            'Add organic amendments regularly',
            'Optimal pH range: 6.5-7.8'
        ],
        'Cotton': [
            'Sowing time: June-July with monsoon',
            'Seed rate: 15-20 kg/ha',
            'Spacing: 90x60 cm',
            'Limited irrigation to prevent waterlogging',
            'Fertilizer: N:P:K - 100:50:50 kg/ha',
            'Integrated pest management essential'
        ],
        'Sugarcane': [
            'Planting season: January-February',
            'Seed rate: 35,000-40,000 setts/ha',
            'Row spacing: 100-120 cm',
            'Careful water management',
            'Fertilizer: N:P:K - 250:100:125 kg/ha',
            'Regular monitoring for pests and diseases'
        ],
        'Jowar': [
            'Sowing time: June-July (Kharif), October (Rabi)',
            'Seed rate: 8-10 kg/ha',
            'Row spacing: 45 cm',
            'Minimal irrigation required',
            'Fertilizer: N:P:K - 80:40:40 kg/ha',
            'Bird control during grain formation'
        ],
        'Wheat': [
            'Sowing time: November',
            'Seed rate: 100 kg/ha',
            'Row spacing: 22.5 cm',
            'Limited irrigation at critical stages',
            'Fertilizer: N:P:K - 120:60:40 kg/ha',
            'Weed management in early stages'
        ],
        'Linseed': [
            'Sowing time: October',
            'Seed rate: 25-30 kg/ha',
            'Row spacing: 30 cm',
            'Minimal irrigation needed',
            'Fertilizer: N:P:K - 40:20:20 kg/ha',
            'Harvesting at physiological maturity'
        ]
    },
    'Clay Soil': {
        'General Practices': [
            'Deep tillage to improve drainage',
            'Add organic matter to improve structure',
            'Avoid working wet soil',
            'Use raised beds for better drainage',
            'Regular soil amendments',
            'Optimal pH range: 6.0-7.0'
        ],
        'Rice': [
            'Puddling for proper field preparation',
            'Seed rate: 25-30 kg/ha',
            'Spacing: 20x15 cm',
            'Proper water management essential',
            'Fertilizer: N:P:K - 120:60:60 kg/ha',
            'Regular monitoring for pests'
        ],
        'Wheat': [
            'Sowing after proper field preparation',
            'Seed rate: 125 kg/ha',
            'Row spacing: 20 cm',
            'Careful irrigation scheduling',
            'Fertilizer: N:P:K - 120:60:40 kg/ha',
            'Disease monitoring important'
        ],
        'Barley': [
            'Sowing time: October-November',
            'Seed rate: 100 kg/ha',
            'Row spacing: 22.5 cm',
            'Limited irrigation needed',
            'Fertilizer: N:P:K - 60:30:30 kg/ha',
            'Harvest at complete maturity'
        ],
        'Oats': [
            'Sowing time: October',
            'Seed rate: 80-100 kg/ha',
            'Row spacing: 20 cm',
            'Irrigation at critical stages',
            'Fertilizer: N:P:K - 80:40:40 kg/ha',
            'Cut at milk stage for fodder'
        ],
        'Potatoes': [
            'Planting time: October-November',
            'Seed rate: 2500-3000 kg/ha',
            'Row spacing: 60x20 cm',
            'Regular irrigation needed',
            'Fertilizer: N:P:K - 120:60:120 kg/ha',
            'Earthing up essential'
        ]
    },
    'Red Soil': {
        'General Practices': [
            'Regular addition of organic matter',
            'Mulching to conserve moisture',
            'Contour farming on slopes',
            'Frequent light irrigation',
            'Soil conservation practices',
            'Optimal pH range: 6.0-6.8'
        ],
        'Groundnut': [
            'Sowing time: June-July',
            'Seed rate: 100-120 kg/ha',
            'Spacing: 30x10 cm',
            'Light but frequent irrigation',
            'Fertilizer: N:P:K - 20:40:40 kg/ha',
            'Gypsum application important'
        ],
        'Pulses': [
            'Season: Kharif and Rabi',
            'Seed rate varies by crop type',
            'Row spacing: 30-45 cm',
            'Minimal irrigation required',
            'Fertilizer: N:P:K - 20:50:20 kg/ha',
            'Rhizobium inoculation beneficial'
        ],
        'Millets': [
            'Sowing with monsoon onset',
            'Seed rate: 8-10 kg/ha',
            'Row spacing: 45 cm',
            'Drought resistant crop',
            'Fertilizer: N:P:K - 40:20:20 kg/ha',
            'Intercropping recommended'
        ],
        'Tobacco': [
            'Transplanting: December-January',
            'Spacing: 90x60 cm',
            'Regular irrigation needed',
            'Fertilizer: N:P:K - 100:50:100 kg/ha',
            'Topping and desuckering essential',
            'Harvest at proper maturity'
        ],
        'Fruits': [
            'Proper spacing by fruit type',
            'Regular basin irrigation',
            'Organic mulching essential',
            'Balanced fertilization',
            'Regular pruning and training',
            'Integrated pest management'
        ]
    }
}

# Soil health records (in-memory storage - replace with database in production)
soil_health_records = []

# Soil analysis standards and thresholds
SOIL_STANDARDS = {
    'nitrogen': {
        'low': {'threshold': 100, 'unit': 'kg/ha'},
        'medium': {'threshold': 200, 'unit': 'kg/ha'},
        'high': {'threshold': float('inf'), 'unit': 'kg/ha'}
    },
    'phosphorus': {
        'low': {'threshold': 20, 'unit': 'kg/ha'},
        'medium': {'threshold': 40, 'unit': 'kg/ha'},
        'high': {'threshold': float('inf'), 'unit': 'kg/ha'}
    },
    'potassium': {
        'low': {'threshold': 150, 'unit': 'kg/ha'},
        'medium': {'threshold': 300, 'unit': 'kg/ha'},
        'high': {'threshold': float('inf'), 'unit': 'kg/ha'}
    },
    'ph': {
        'acidic': {'threshold': 6.0, 'unit': 'pH'},
        'neutral': {'threshold': 7.0, 'unit': 'pH'},
        'alkaline': {'threshold': float('inf'), 'unit': 'pH'}
    },
    'organic_matter': {
        'low': {'threshold': 2, 'unit': '%'},
        'medium': {'threshold': 4, 'unit': '%'},
        'high': {'threshold': float('inf'), 'unit': '%'}
    }
}

# Crop-specific nutrient requirements
CROP_REQUIREMENTS = {
    'Rice': {
        'nitrogen': {'min': 120, 'max': 180},
        'phosphorus': {'min': 25, 'max': 40},
        'potassium': {'min': 160, 'max': 200},
        'ph': {'min': 5.5, 'max': 7.0}
    },
    'Wheat': {
        'nitrogen': {'min': 100, 'max': 150},
        'phosphorus': {'min': 20, 'max': 35},
        'potassium': {'min': 150, 'max': 180},
        'ph': {'min': 6.0, 'max': 7.5}
    },
    # Add more crops as needed
}

# Chat history storage
chat_history = []

# Initialize the soil health analyzer
soil_analyzer = SoilHealthAnalyzer(
    model_path=os.path.join('models', 'soil_ph_model.joblib'),
    scaler_path=os.path.join('models', 'soil_ph_scaler.joblib')
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def get_current_season():
    month = datetime.now().month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/soil-classification', methods=['GET', 'POST'])
def soil_classification():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        # Save and process image
        file_path = os.path.join('static', 'uploads', file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        try:
            img_array = preprocess_image(file_path)
            prediction = soil_model.predict(img_array)
            predicted_class = SOIL_LABELS[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            
            # Store results in session
            session['soil_type'] = predicted_class
            session['soil_confidence'] = confidence
            
            return render_template('soil_result.html',
                                 soil_type=predicted_class,
                                 confidence=confidence,
                                 image_path=file_path)
        except Exception as e:
            flash(f'Error processing image: {str(e)}')
            return redirect(request.url)
    
    return render_template('soil_classification.html')

@app.route('/crop-recommendations')
def crop_recommendations():
    soil_type = session.get('soil_type')
    if not soil_type:
        flash('Please classify your soil first')
        return redirect(url_for('soil_classification'))
    
    recommended_crops = CROP_RECOMMENDATIONS.get(soil_type, [])
    return render_template('crop_recommendations.html',
                         soil_type=soil_type,
                         crops=recommended_crops)

@app.route('/cultivation_practices/<soil_type>')
def cultivation_practices(soil_type):
    # Convert soil_type to proper format (e.g., 'black' to 'Black Soil')
    soil_type = f"{soil_type.title()} Soil"
    
    if soil_type in CULTIVATION_PRACTICES:
        practices = CULTIVATION_PRACTICES[soil_type]
        crops = CROP_RECOMMENDATIONS[soil_type]
        return render_template('cultivation_practices.html', 
                            soil_type=soil_type,
                            practices=practices,
                            crops=crops)
    else:
        flash('Invalid soil type selected.', 'error')
        return redirect(url_for('home'))

@app.route('/plant-disease')
def plant_disease():
    return render_template('plant_disease.html')

@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a PNG or JPG image.'}), 400
    
    try:
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Check if model is loaded
        if plant_disease_model is None:
            return jsonify({'error': 'Model not loaded. Please try again later.'}), 500
        
        # Preprocess image
        img_array = preprocess_image(filepath)
        if img_array is None:
            return jsonify({'error': 'Error processing image. Please try a different image.'}), 400
        
        # Make prediction
        prediction = plant_disease_model.predict(img_array)
        predicted_class = PLANT_DISEASE_LABELS[np.argmax(prediction[0])]
        
        # Clean up
        if os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({'prediction': predicted_class})
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        # Clean up in case of error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': 'An error occurred during prediction. Please try again.'}), 500

def analyze_soil_health(image_path):
    """Analyze soil health from image using ML model"""
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)
        
        # Get soil type prediction
        soil_prediction = soil_model.predict(img_array)
        soil_type = SOIL_LABELS[np.argmax(soil_prediction)]
        confidence = float(np.max(soil_prediction))
        
        # Calculate health score based on soil type and characteristics
        # This is a simplified version - in a real application, you would use
        # more sophisticated analysis based on soil type and characteristics
        base_score = 5  # Base score
        type_multiplier = {
            'Alluvial Soil': 1.2,
            'Black Soil': 1.1,
            'Clay Soil': 1.0,
            'Red Soil': 0.9
        }
        
        health_score = base_score * type_multiplier.get(soil_type, 1.0)
        health_score = min(10, max(5, health_score))  # Normalize between 5-10
        
        return {
            'soil_type': soil_type,
            'confidence': confidence,
            'health_score': health_score,
            'status': 'Good' if health_score >= 7 else 'Needs Improvement'
        }
    except Exception as e:
        print(f"Error analyzing soil health: {e}")
        return None

def assess_soil_fertility(nitrogen, phosphorus, potassium, ph, organic_matter, crop_type=None):
    """Assess soil fertility based on nutrient levels and standards"""
    results = {
        'nutrients': {},
        'overall_score': 0,
        'recommendations': []
    }
    
    # Analyze each nutrient
    for nutrient, value in [('nitrogen', nitrogen), ('phosphorus', phosphorus), 
                          ('potassium', potassium)]:
        standards = SOIL_STANDARDS[nutrient]
        level = 'low'
        if value >= standards['high']['threshold']:
            level = 'high'
        elif value >= standards['medium']['threshold']:
            level = 'medium'
            
        results['nutrients'][nutrient] = {
            'value': value,
            'level': level,
            'unit': standards['low']['unit']
        }
        
        # Add to overall score
        if level == 'high':
            results['overall_score'] += 3
        elif level == 'medium':
            results['overall_score'] += 2
        else:
            results['overall_score'] += 1
    
    # Analyze pH
    if ph < SOIL_STANDARDS['ph']['acidic']['threshold']:
        ph_level = 'acidic'
    elif ph <= SOIL_STANDARDS['ph']['neutral']['threshold']:
        ph_level = 'neutral'
    else:
        ph_level = 'alkaline'
    
    results['nutrients']['ph'] = {
        'value': ph,
        'level': ph_level,
        'unit': 'pH'
    }
    
    # Analyze organic matter
    om_standards = SOIL_STANDARDS['organic_matter']
    om_level = 'low'
    if organic_matter >= om_standards['high']['threshold']:
        om_level = 'high'
    elif organic_matter >= om_standards['medium']['threshold']:
        om_level = 'medium'
    
    results['nutrients']['organic_matter'] = {
        'value': organic_matter,
        'level': om_level,
        'unit': '%'
    }
    
    # Generate recommendations
    if crop_type and crop_type in CROP_REQUIREMENTS:
        crop_req = CROP_REQUIREMENTS[crop_type]
        for nutrient in ['nitrogen', 'phosphorus', 'potassium']:
            value = results['nutrients'][nutrient]['value']
            if value < crop_req[nutrient]['min']:
                results['recommendations'].append(
                    f"Add {nutrogen} fertilizer to meet {crop_type} requirements"
                )
            elif value > crop_req[nutrient]['max']:
                results['recommendations'].append(
                    f"Reduce {nutrient} application for {crop_type}"
                )
    
    # Add general recommendations
    if ph_level != 'neutral':
        results['recommendations'].append(
            f"Adjust pH to neutral range (6.0-7.0) using {'lime' if ph_level == 'acidic' else 'sulfur'}"
        )
    
    if om_level == 'low':
        results['recommendations'].append("Add organic compost to improve soil structure")
    
    # Calculate final status
    if results['overall_score'] >= 12:
        results['status'] = 'Excellent'
        results['status_class'] = 'success'
    elif results['overall_score'] >= 9:
        results['status'] = 'Good'
        results['status_class'] = 'warning'
    else:
        results['status'] = 'Needs Improvement'
        results['status_class'] = 'danger'
    
    return results

@app.route('/soil-health', methods=['GET', 'POST'])
def soil_health():
    """
    Handle both the soil health page rendering and soil analysis.
    """
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)
        
        # Save the uploaded file
        file_path = os.path.join('static', 'uploads', file.filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file.save(file_path)
        
        try:
            # Create a temporary image to get RGB values
            img = Image.open(file_path)
            # Get RGB values from the center of the image
            width, height = img.size
            center_x = width // 2
            center_y = height // 2
            rgb_values = img.getpixel((center_x, center_y))[:3]  # Get RGB values
            
            # Analyze soil health using the model
            result = soil_analyzer.analyze_soil_health(rgb_values)
            
            # Add record to storage
            soil_health_records.append({
                'date': datetime.now().strftime('%Y-%m-%d'),
                'location': request.form.get('location', 'Unknown'),
                'ph': result['soil_ph'],
                'status': result['health_status'],
                'image_path': file_path
            })
            
            return jsonify(result)
            
        except Exception as e:
            return jsonify({
                'error': str(e),
                'soil_ph': None,
                'health_status': 'Unknown',
                'health_description': 'An error occurred during analysis',
                'suitable_crops': [],
                'improvement_recommendations': ['Please try again or contact support']
            }), 500
            
    # GET request - render the template
    return render_template('soil_health.html', soil_records=soil_health_records)

@app.route('/seasonal-guide')
def seasonal_guide():
    """
    Render the seasonal guide page with dropdown-based recommendations.
    """
    return render_template('seasonal_guide.html')

@app.route('/crop-rotation')
def crop_rotation():
    return render_template('crop_rotation.html')

@app.route('/fertilizer-recommendation')
def fertilizer_recommendation():
    return render_template('fertilizer_recommendation.html')

@app.route('/organic-farming')
def organic_farming():
    return render_template('organic_farming.html')

@app.route('/chatbot', methods=['GET'])
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot-api', methods=['POST'])
def chatbot_api():
    try:
        print("Received chatbot API request")  # Debug log
        data = request.get_json()
        
        # Extract message from the new frontend format
        contents = data.get('contents', [])
        if not contents or not isinstance(contents, list):
            return jsonify({'error': 'Invalid request format'}), 400
            
        user_message = contents[0].get('parts', [{}])[0].get('text', '').strip()
        
        print(f"User input: {user_message}")  # Debug log
        
        if not user_message:
            return jsonify({'error': 'No input provided'}), 400
            
        if not model:
            return jsonify({'error': 'Chatbot service is not available'}), 503
            
        try:
            # Create agricultural context
            prompt = f"""You are an agricultural expert chatbot. Please provide helpful advice about:
            - Soil types and classification
            - Crop recommendations
            - Soil health monitoring
            - Fertilizer recommendations
            - Plant diseases
            - Farming practices
            - Seasonal planting

            Format your response with:
            - Use bullet points for lists
            - Bold important terms using **term**
            - Keep responses concise and focused
            - Provide practical, actionable advice

            User question: {user_message}"""

            print("Generating response...")  # Debug log
            
            # Generate response using the model with retry
            @retry.Retry(predicate=retry.if_exception_type(Exception))
            def generate_with_retry():
                return model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            
            response = generate_with_retry()
            
            print(f"Raw response: {response}")  # Debug log
            
            if not response or not hasattr(response, 'text'):
                return jsonify({
                    'candidates': [{
                        'content': {
                            'parts': [{'text': 'I apologize, but I am unable to process your request at the moment.'}]
                        }
                    }]
                }), 200

            # Clean and format the response
            reply = response.text.strip()
            reply = reply.replace('AI:', '').replace('Assistant:', '').strip()
            
            # Format response to match frontend expectations
            return jsonify({
                'candidates': [{
                    'content': {
                        'parts': [{'text': reply}]
                    }
                }]
            }), 200
                
        except Exception as e:
            print(f"Model error: {str(e)}")
            print(traceback.format_exc())
            return jsonify({
                'candidates': [{
                    'content': {
                        'parts': [{'text': f'I apologize, but I encountered an error: {str(e)}'}]
                    }
                }]
            }), 200
            
    except Exception as e:
        print(f"Error in chatbot API: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'candidates': [{
                'content': {
                    'parts': [{'text': 'An error occurred while processing your request.'}]
                }
            }]
        }), 200

@app.route('/analyze_soil_health', methods=['POST'])
def analyze_soil_health():
    try:
        data = request.get_json()
        rgb_values = data.get('rgb_values')
        
        if not rgb_values or len(rgb_values) != 3:
            return jsonify({
                'error': 'Invalid RGB values provided',
                'soil_ph': None,
                'health_status': 'Unknown',
                'health_description': 'Could not determine soil health',
                'suitable_crops': [],
                'improvement_recommendations': ['Please provide a valid soil image']
            }), 400
        
        # Analyze soil health using the model
        result = soil_analyzer.analyze_soil_health(rgb_values)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'soil_ph': None,
            'health_status': 'Unknown',
            'health_description': 'An error occurred during analysis',
            'suitable_crops': [],
            'improvement_recommendations': ['Please try again or contact support']
        }), 500

@app.route('/flower-analysis')
def flower_analysis():
    return render_template('flower_analysis.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze-flower', methods=['POST'])
def analyze_flower():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a PNG or JPG image.'}), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Error reading image file'}), 400
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Enhance texture
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced = cv2.filter2D(gray, -1, kernel)
        
        # Simulate UV effect
        uv = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Detect veins
        edges = cv2.Canny(gray, 100, 200)
        
        # Create collage
        # Resize all images to the same height
        height = gray.shape[0]
        width = gray.shape[1]
        
        # Resize all images to the same dimensions
        gray = cv2.resize(gray, (width, height))
        enhanced = cv2.resize(enhanced, (width, height))
        uv = cv2.resize(uv, (width, height))
        edges = cv2.resize(edges, (width, height))
        
        # Convert single channel images to 3 channels for concatenation
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        enhanced_3ch = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Create collage
        collage = np.hstack((gray_3ch, enhanced_3ch, uv, edges_3ch))
        
        # Save the result
        result_filename = f'result_{filename}'
        result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
        cv2.imwrite(result_path, collage)
        
        # Clean up the original file
        os.remove(filepath)
        
        # Return the URL for the result image
        return jsonify({
            'result_image': url_for('uploaded_file', filename=result_filename)
        })
        
    except Exception as e:
        print(f"Error in flower analysis: {str(e)}")
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': f'An error occurred during analysis: {str(e)}'}), 500

@app.route('/soil-explorer')
def soil_explorer():
    return render_template('soil_explorer.html')

if __name__ == '__main__':
    app.run(debug=True) 