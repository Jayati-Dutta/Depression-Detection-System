import logging
import joblib

logging.basicConfig(level=logging.INFO)

# Load model, scaler, and features once (on import)
logging.info("Loading quiz pipeline...")
pipeline_data = joblib.load('quiz_pipeline.pkl')
model = pipeline_data['model']
scaler = pipeline_data['scaler']
features = pipeline_data['features']  # list of quiz question names

def predict_depression_level(user_inputs):
    logging.info("Scaling user input...")
    user_inputs_scaled = scaler.transform([user_inputs])

    logging.info("Predicting depression level...")
    depression_level = model.predict(user_inputs_scaled)[0]

    return int(depression_level)

def get_quiz_questions():
    return features
