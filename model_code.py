# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import joblib

# ----------------------------
# 1. Load and Prepare Data
# ----------------------------
df = pd.read_csv("Final_Model_Training_Data.csv")
df.columns = df.columns.str.strip()

# Fill missing values
df = df.fillna({
    'Seat_Multiplier': 3.0,
    'Stage': 'Unknown',
    'Venue': 'Unknown',
    'Team_1': 'Unknown',
    'Team_2': 'Unknown'
})
df['Base_Price_Base'] = df['Base_Price_Base'].fillna(df['Base_Price'] / df['Seat_Multiplier'])

# ----------------------------
# 2. Encode Categorical Columns (FIXED: Using OrdinalEncoder)
# ----------------------------
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
categorical_cols = ['Importance', 'Stage', 'Venue', 'Team_1', 'Team_2']
df[['Importance_Num', 'Stage_Num', 'Venue_Num', 'Team1_Num', 'Team2_Num']] = encoder.fit_transform(df[categorical_cols])

# ----------------------------
# 3. Train Model
# ----------------------------
X = df[['Base_Price_Base', 'Importance_Num', 'Stage_Num', 'Venue_Num',
        'Team1_Num', 'Team2_Num', 'Days_until_match', 'Tickets_Sold', 'Year']]
y = df['Base_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# ----------------------------
# 4. Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

# ----------------------------
# 5. Save Model and Encoder (FIXED: Single encoder for all categories)
# ----------------------------
joblib.dump(model, "final_random_forest_model.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")  # Save the encoder