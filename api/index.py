from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from vercel_python.wsgi import VercelWSGIHandler  # For Vercel deployment

app = Flask(__name__)

# Constants
REQUIRED_FIELDS = [
    'base_price', 'seat_multiplier', 'tickets_sold', 'days_until',
    'importance', 'stage', 'venue', 'team1', 'team2', 'year'
]

# Load model and encoder
try:
    model = joblib.load("final_random_forest_model.pkl")
    encoder = joblib.load("ordinal_encoder.pkl")
except Exception as e:
    print(f"Error loading model files: {e}")
    # Handle this appropriately in production (maybe exit or notify)

@app.route("/")
def index():
    return render_template("index.html", price=None)

@app.route("/match")
def match():
    return render_template("match.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        missing_fields = [field for field in REQUIRED_FIELDS if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'missing_fields': missing_fields
            }), 400

        # Convert and validate numerical fields
        try:
            base_price = float(data['base_price'])
            seat_multiplier = float(data['seat_multiplier'])
            tickets_sold = int(data['tickets_sold'])
            days_until = int(data['days_until'])
            year = int(data['year'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'error': 'Invalid numerical values',
                'details': str(e)
            }), 400

        # Encode categorical features
        try:
            encoded_features = encoder.transform([[
                data['importance'], 
                data['stage'], 
                data['venue'], 
                data['team1'], 
                data['team2']
            ]])[0]
            importance_num, stage_num, venue_num, team1_num, team2_num = encoded_features
        except Exception as e:
            return jsonify({
                'error': 'Error encoding categorical features',
                'details': str(e)
            }), 400

        # Prepare prediction data
        base_price_base = base_price / seat_multiplier
        prediction_data = pd.DataFrame([{
            "Base_Price_Base": base_price_base,
            "Importance_Num": importance_num,
            "Stage_Num": stage_num,
            "Venue_Num": venue_num,
            "Team1_Num": team1_num,
            "Team2_Num": team2_num,
            "Days_until_match": days_until,
            "Tickets_Sold": tickets_sold,
            "Year": year
        }])

        # Predict
        try:
            predicted_price = model.predict(prediction_data)[0]
            return jsonify({
                'predicted_price': round(predicted_price, 2),
                'status': 'success'
            })
        except Exception as e:
            return jsonify({
                'error': 'Prediction failed',
                'details': str(e)
            }), 500

    except Exception as e:
        print(f"Unexpected error: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

# Vercel handler
handler = VercelWSGIHandler(app)

if __name__ == "__main__":
    app.run(debug=True)