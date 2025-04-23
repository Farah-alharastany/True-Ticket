from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and encoder (FIXED: Single encoder)
model = joblib.load("final_random_forest_model.pkl")
encoder = joblib.load("ordinal_encoder.pkl")  # Load the encoder

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

        # Extract and validate fields
        required_fields = ['base_price', 'seat_multiplier', 'tickets_sold', 'days_until', 
                          'importance', 'stage', 'venue', 'team1', 'team2', 'year']
        if not all(data.get(field) for field in required_fields):
            return jsonify({'error': 'يرجى ملء جميع الحقول بشكل صحيح.'}), 400

        # Convert values
        base_price = float(data['base_price'])
        seat_multiplier = float(data['seat_multiplier'])
        tickets_sold = int(data['tickets_sold'])
        days_until = int(data['days_until'])
        year = int(data['year'])

        # Encode categorical features (FIXED: Use the single encoder)
        encoded_features = encoder.transform([[data['importance'], data['stage'], 
                                             data['venue'], data['team1'], data['team2']]])[0]
        importance_num, stage_num, venue_num, team1_num, team2_num = encoded_features

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
        print("Encoded features:", encoded_features)
        print("Features sent to model:\n", prediction_data)
        # Predict
        predicted_price = model.predict(prediction_data)[0]
        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': f'حدث خطأ أثناء المعالجة: {str(e)}'}), 500

if __name__ == "__main__":
    app.run(debug=True)