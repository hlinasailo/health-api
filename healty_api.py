# health_api.py
# Unified Flask API for Multiple Disease Predictions
# Includes: Diabetes, Stroke, Heart Disease, and Lung Disease

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import traceback

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====================================================
# 1️⃣ Model Loader Helper Function
# ====================================================
def load_model(filename):
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        print(f"❌ Model file not found: {path}")
        return None
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            return data
        return {'model': data}
    except Exception as e:
        print(f"❌ Failed to load {filename}: {e}")
        return None


# ====================================================
# 2️⃣ Load All Models
# ====================================================
models = {
    'diabetes': load_model('diabetes_model.pkl'),
    'stroke': load_model('stroke_model.pkl'),
    'heart': load_model('heart_disease_model.pkl'),
    'lung': load_model('lung_disease_model.pkl')
}

print("✅ Model loading summary:")
for name, m in models.items():
    print(f"  {name}: {'Loaded' if m else 'Missing'}")


# ====================================================
# 3️⃣ Diabetes Prediction
# ====================================================
@app.route('/diabetes-predict', methods=['POST'])
def predict_diabetes():
    model_data = models.get('diabetes')
    if not model_data or 'model' not in model_data:
        return jsonify({'error': 'Diabetes model not loaded'}), 500

    try:
        data = request.get_json()
        full_fields = ['pregnancies', 'glucose', 'blood_pressure', 'skin_thickness',
                       'insulin', 'bmi', 'diabetes_pedigree_function', 'age']
        reduced_fields = ['glucose', 'blood_pressure', 'bmi',
                          'diabetes_pedigree_function', 'age']

        if all(k in data for k in full_fields):
            df = pd.DataFrame([{
                'Pregnancies': float(data['pregnancies']),
                'Glucose': float(data['glucose']),
                'BloodPressure': float(data['blood_pressure']),
                'SkinThickness': float(data['skin_thickness']),
                'Insulin': float(data['insulin']),
                'BMI': float(data['bmi']),
                'DiabetesPedigreeFunction': float(data['diabetes_pedigree_function']),
                'Age': int(data['age'])
            }])
        elif all(k in data for k in reduced_fields):
            df = pd.DataFrame([{
                'Glucose': float(data['glucose']),
                'BloodPressure': float(data['blood_pressure']),
                'BMI': float(data['bmi']),
                'DiabetesPedigreeFunction': float(data['diabetes_pedigree_function']),
                'Age': int(data['age'])
            }])
        else:
            return jsonify({'error': 'Missing or invalid fields'}), 400

        model = model_data['model']
        prob = float(model.predict_proba(df)[0][1]) if hasattr(model, 'predict_proba') else float(model.predict(df)[0])
        pred = int(prob >= 0.5)
        message = "⚠️ High diabetes risk. Consult a doctor." if pred == 1 else "✅ Low diabetes risk. Stay healthy!"

        return jsonify({'prediction': pred, 'probability': prob, 'message': message})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ====================================================
# 4️⃣ Stroke Prediction
# ====================================================
@app.route('/stroke-predict', methods=['POST'])
def predict_stroke():
    model_data = models.get('stroke')
    if not model_data or 'model' not in model_data:
        return jsonify({'error': 'Stroke model not loaded'}), 500

    try:
        data = request.get_json()
        model = model_data['model']
        le_gender = model_data['le_gender']
        le_married = model_data['le_married']
        le_work = model_data['le_work']
        le_residence = model_data['le_residence']
        le_smoke = model_data['le_smoke']

        input_df = pd.DataFrame([{
            'age': data['age'],
            'hypertension': data['hypertension'],
            'heart_disease': data['heart_disease'],
            'avg_glucose_level': data['avg_glucose_level'],
            'bmi': data['bmi'],
            'gender': le_gender.transform([data['gender']])[0],
            'ever_married': le_married.transform([data['ever_married']])[0],
            'work_type': le_work.transform([data['work_type']])[0],
            'residence_type': le_residence.transform([data['residence_type']])[0],
            'smoking_status': le_smoke.transform([data['smoking_status']])[0],
        }])

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])
        message = "⚠️ High risk of stroke detected. Please consult a doctor." if pred == 1 else "✅ Low stroke risk."

        return jsonify({'prediction': pred, 'probability': prob, 'message': message})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ====================================================
# 5️⃣ Heart Disease Prediction
# ====================================================
@app.route('/heart-predict', methods=['POST'])
def predict_heart():
    model_data = models.get('heart')
    if not model_data or 'model' not in model_data:
        return jsonify({'error': 'Heart model not loaded'}), 500

    try:
        data = request.get_json()
        model = model_data['model']
        le_sex = model_data['le_sex']

        input_df = pd.DataFrame([{
            'age': data['age'],
            'sex': le_sex.transform([data['sex']])[0] if isinstance(data['sex'], str) else data['sex'],
            'cp': data['cp'],
            'trestbps': data['trestbps'],
            'chol': data['chol'],
            'thalach': data['thalach'],
            'exang': data['exang'],
            'oldpeak': data['oldpeak']
        }])

        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])
        message = "⚠️ High heart disease risk. Consult a cardiologist." if pred == 1 else "✅ Low heart disease risk."

        return jsonify({'prediction': pred, 'probability': prob, 'message': message})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ====================================================
# 6️⃣ Lung Disease Prediction
# ====================================================
@app.route('/lung-disease-predict', methods=['POST'])
def predict_lung_disease():
    model_data = models.get('lung')
    if not model_data:
        return jsonify({'error': 'Lung disease model not loaded'}), 500

    try:
        model = model_data['model']
        le_gender = model_data['le_gender']
        le_smoke = model_data['le_smoke']
        feature_columns = model_data['feature_columns']

        data = request.get_json()
        required_fields = ['age', 'gender', 'smoking_history', 'cough', 'shortness_of_breath', 'chest_pain', 'wheezing']
        missing = [k for k in required_fields if k not in data]
        if missing:
            return jsonify({'error': 'Missing required fields', 'missing': missing}), 400

        # Encode categorical features using saved LabelEncoders
        df = pd.DataFrame([[
            int(data['age']),
            le_gender.transform([data['gender']])[0],
            le_smoke.transform([data['smoking_history']])[0],
            int(data['cough']),
            int(data['shortness_of_breath']),
            int(data['chest_pain']),
            int(data['wheezing'])
        ]], columns=feature_columns)

        prob = float(model.predict_proba(df)[0][1])
        pred = int(prob >= 0.5)

        # Determine disease type for message
        disease_type = "Respiratory Condition"
        if data.get('smoking_history', '').lower() == 'current' and pred == 1:
            disease_type = "Potential COPD"
        elif str(data.get('wheezing', '')).lower() in ('yes','true','1') and pred == 1:
            disease_type = "Potential Asthma"

        message = f"⚠️ {disease_type} detected. Consult a pulmonologist." if pred == 1 else "✅ Healthy lungs detected. Maintain respiratory health."

        return jsonify({
            'prediction': pred,
            'probability': round(prob,3),
            'disease_type': disease_type,
            'message': message
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500




# ====================================================
# 7️⃣ Health Check Route
# ====================================================
@app.route('/')
def home():
    return jsonify({
        'status': 'API is running',
        'models_loaded': {k: bool(v) for k, v in models.items()},
        'endpoints': {
            'diabetes': '/diabetes-predict',
            'stroke': '/stroke-predict',
            'heart': '/heart-predict',
            'lung': '/lung-disease-predict'
        }
    })


# ====================================================
# 8️⃣ Run Flask Server
# ====================================================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
