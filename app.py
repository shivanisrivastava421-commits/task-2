import numpy as np
import pickle
from flask import Flask, render_template, request
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Attempt to import the custom NaiveBayes class
    from naive_bayes import NaiveBayes
    print("✅ DEBUG: Successfully found and imported NaiveBayes class definition.") # <--- ADD THIS LINE
except ImportError:
    print("FATAL ERROR: Could not import NaiveBayes class. Ensure naive_bayes.py is in the root directory.")
    
sys.path.pop(0)
# Initialize Flask application
app = Flask(__name__)

# --- Model Loading ---
MODEL_PATH = 'model.pkl'
model = None

try:
    with open(MODEL_PATH, 'rb') as file:
        model = pickle.load(file)
    print("✅ SERVER START: Model loaded successfully from model.pkl")
except Exception as e:
    print(f"❌ SERVER ERROR: Error loading model: {e}")

# --- Utility Function for Prediction ---

def prepare_and_predict(form_data):
    """Prepares form data into a feature vector and makes a prediction."""
    
    if model is None:
        return "Model failed to load on server startup. Cannot make prediction.", False

    # 1. Extract and convert numerical fields to float (10 features total expected by model)
    try:
        # Note: Student ID is excluded from features
        cgpa = float(form_data.get('cgpa'))
        internships = float(form_data.get('internships'))
        projects = float(form_data.get('projects'))
        workshops = float(form_data.get('workshops'))
        aptitude = float(form_data.get('aptitude_score'))
        soft_skills = float(form_data.get('soft_skills'))
        ssc_marks = float(form_data.get('ssc_marks'))
        hsc_marks = float(form_data.get('hsc_marks'))
    except (ValueError, TypeError):
        return "Input Error: Please ensure all required numerical fields are correctly filled.", False

    # 2. Convert categorical (Yes/No) fields to 0 or 1
    extracurricular = 1.0 if form_data.get('extracurricular') == 'Yes' else 0.0
    placement_training = 1.0 if form_data.get('placement_training') == 'Yes' else 0.0
    
    # 3. Assemble the feature vector (10 features)
    features = [
        cgpa, 
        internships, 
        projects, 
        workshops, 
        aptitude, 
        soft_skills, 
        extracurricular, 
        placement_training,
        ssc_marks, 
        hsc_marks
    ]

    final_features = np.array(features).reshape(1, -1)

    # 4. Make prediction
    try:
        prediction = model.predict(final_features)[0]
        # Assuming 1 = Placed, 0 = Not Placed
        result_text = "The predicted outcome is: PLACED 🎉" if prediction == 1 else "The predicted outcome is: NOT PLACED 😔"
        return result_text, True
    except Exception as e:
        print(f"Prediction logic error: {e}")
        return "An internal prediction error occurred. Check server logs.", False

# --- Flask Routes ---

@app.route('/', methods=['GET'])
def home():
    """THIS ROUTE RENDERS THE HTML FRONTEND (index.html)."""
    print("--- 🌐 REQUEST: Serving HTML Frontend (index.html) ---") 
    return render_template('index.html', prediction_text="", error=False)

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, makes prediction, and renders result."""
    
    if model is None:
        return render_template('index.html', prediction_text="Error: Model failed to load on server startup.", error=True)

    result, success = prepare_and_predict(request.form)
    
    return render_template('index.html', prediction_text=result, error=(not success))

# --- Run the Application ---

if __name__ == '__main__':
    # *** CRITICAL: Running on Port 8000 to bypass the conflict on Port 5000 ***
    app.run(debug=True, port=8000)



