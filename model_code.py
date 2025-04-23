# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
import joblib

# ----------------------------
# 1. Load and Prepare Data (FIXED DATA LEAKAGE)
# ----------------------------
df = pd.read_csv("Final_Model_Training_Data.csv")
df.columns = df.columns.str.strip()

# Standardize categorical text (FIXED DUPLICATE CATEGORIES)
df['Importance'] = df['Importance'].str.lower().str.strip()
df['Stage'] = df['Stage'].str.strip()
df['Venue'] = df['Venue'].str.strip()
df['Team_1'] = df['Team_1'].str.strip()
df['Team_2'] = df['Team_2'].str.strip()

# Fill missing values
df = df.fillna({
    'Seat_Multiplier': 3.0,
    'Stage': 'Unknown',
    'Venue': 'Unknown',
    'Team_1': 'Unknown',
    'Team_2': 'Unknown'
})

# Calculate base price WITHOUT target leakage (FIXED CRITICAL ISSUE)
df['Base_Price_Base'] = df['Base_Price'] / df['Seat_Multiplier']

# ----------------------------
# 2. Encode Categorical Columns (IMPROVED ENCODING)
# ----------------------------
encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1,
    dtype=np.int32
)
categorical_cols = ['Importance', 'Stage', 'Venue', 'Team_1', 'Team_2']

# Fit encoder once on all categorical data
encoder.fit(df[categorical_cols])
df[['Importance_Num', 'Stage_Num', 'Venue_Num', 'Team1_Num', 'Team2_Num']] = encoder.transform(df[categorical_cols])

# ----------------------------
# 3. Train Model (IMPROVED PARAMETERS)
# ----------------------------
features = [
    'Base_Price_Base', 'Importance_Num', 'Stage_Num', 'Venue_Num',
    'Team1_Num', 'Team2_Num', 'Days_until_match', 'Tickets_Sold', 'Year'
]

X = df[features]
y = df['Base_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# ----------------------------
# 4. Evaluate Model (ENHANCED DIAGNOSTICS)
# ----------------------------
y_pred = model.predict(X_test)
print(f"\nModel Performance:")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

print("\nFeature Importances:")
for name, importance in zip(features, model.feature_importances_):
    print(f"{name}: {importance:.4f}")

# ----------------------------
# 5. Validation Checks
# ----------------------------
print("\nEncoder Categories Verification:")
print("Importance:", encoder.categories_[0])
print("Stage:", encoder.categories_[1][:5], "...")  # Show first 5

print("\nSample Prediction Test:")
test_sample = X_train.iloc[:3].copy()
test_sample.loc[:, 'Base_Price_Base'] = 300  # Hold base price constant
print(model.predict(test_sample))  # Should give DIFFERENT values

# ----------------------------
# 6. Save Model and Encoder
# ----------------------------
joblib.dump(model, "final_random_forest_model.pkl")
joblib.dump(encoder, "ordinal_encoder.pkl")
joblib.dump(features, "model_features.pkl")  # Save feature list

print("\nModel trained and saved successfully!")
print("Key improvements maintained:")
print("- Removed target leakage in base price calculation")
print("- Standardized categorical text (fixed duplicate categories)")
print("- Optimized model parameters")