import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os

# -------------------------
# Download Dataset if Missing
DATA_URL = "https://raw.githubusercontent.com/amrrs/crop-recommendation-dataset/main/Crop_recommendation.csv"
DATA_FILE = "Crop_recommendation.csv"

import urllib.request
urllib.request.urlretrieve(DATA_URL, DATA_FILE)


if not os.path.exists(DATA_FILE):
    import urllib.request
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print("Dataset downloaded.")

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv(DATA_FILE)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

# Encode target
le = LabelEncoder()
y_enc = le.fit_transform(y)

# Train model
model = GradientBoostingClassifier()
model.fit(X, y_enc)

# Save model and encoder
dump(model, "crop_model.joblib")
dump(le, "label_encoder.joblib")

print("✅ Model and label encoder saved successfully.")
