import streamlit as st
import requests
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -------------------
# Streamlit config
# -------------------
st.set_page_config(page_title="Crop Prediction Assistant", page_icon="🌱")
st.title("🌾 AI Crop Recommendation Assistant")
st.markdown("Farm assistant that explains **ML-based crop predictions** using AI language model.")

# -------------------
# Load environment variables
# -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Set it in .env or Streamlit secrets.")
    st.stop()

# -------------------
# Detect user location automatically
# -------------------
def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        loc = res.json()["loc"].split(",")
        return float(loc[0]), float(loc[1])
    except:
        # Default fallback
        return 28.61, 77.20  

lat, lon = get_user_location()
st.success(f"📍 Detected Location: {lat:.3f}, {lon:.3f}")

# -------------------
# Fetch soil data from SoilGrids API
# -------------------
def fetch_soil_data(lat, lon):
    url = f"https://rest.isric.org/soilgrids/query?lon={lon}&lat={lat}&attributes=phh2o,nitrogen,ocd,sand,silt"
    r = requests.get(url)
    if r.status_code != 200:
        return {"phh2o": 6.5, "nitrogen": 50, "ocd": 10, "sand": 40, "silt": 40}  # fallback
    data = r.json()
    properties = data.get("properties", {})
    # Extract topsoil (0-5 cm) mean values
    soil_dict = {}
    for attr, attr_data in properties.items():
        soil_dict[attr] = attr_data.get("M", {}).get("0-5cm", 0)
    return soil_dict

soil_data = fetch_soil_data(lat, lon)
st.write("🧪 Soil Data")
st.json(soil_data)

# -------------------
# Fetch weather data
# -------------------
def fetch_weather(lat, lon):
    if not WEATHER_API_KEY:
        return {"temp_c": 25, "humidity": 70, "precip_mm": 2, "wind_kph": 10}
    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
    r = requests.get(url)
    if r.status_code != 200:
        return {"temp_c": 25, "humidity": 70, "precip_mm": 2, "wind_kph": 10}
    current = r.json().get("current", {})
    return {
        "temp_c": current.get("temp_c", 25),
        "humidity": current.get("humidity", 70),
        "precip_mm": current.get("precip_mm", 2),
        "wind_kph": current.get("wind_kph", 10)
    }

weather_data = fetch_weather(lat, lon)
st.write("🌤 Weather Data")
st.json(weather_data)

# -------------------
# Prepare features for ML
# -------------------
def prepare_features(soil_dict, weather_dict):
    soil_df = pd.DataFrame([soil_dict])
    weather_df = pd.DataFrame([weather_dict])
    full_df = pd.concat([soil_df, weather_df], axis=1)
    full_df = full_df.fillna(full_df.mean(numeric_only=True))
    features = full_df.select_dtypes(include=["float64", "int64"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, features.columns

X_scaled, feature_names = prepare_features(soil_data, weather_data)

# -------------------
# Train dummy ML model
# -------------------
def train_model(X):
    np.random.seed(42)
    y = np.random.choice([0, 1, 2], size=X.shape[0])
    if X.shape[0] == 1:
        X = np.tile(X, (20, 1))
        y = np.random.choice([0, 1, 2], size=X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

clf = train_model(X_scaled)

# -------------------
# Predict crop
# -------------------
crop_map = {0: "🌾 Wheat", 1: "🌱 Rice", 2: "🌽 Maize"}
predicted_crop = crop_map.get(clf.predict(X_scaled)[0], "Unknown")
st.success(f"✅ ML Model Suggestion: {predicted_crop}")

# -------------------
# Groq LLM setup
# -------------------
MODEL_NAME = "gemma2-9b-it"
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name=MODEL_NAME,
    temperature=0.7,
    streaming=True
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an Agriculture AI Assistant. Explain crop recommendations "
               "based on soil + weather + ML model output in very simple farmer-friendly language. "
               "Give reasons why a crop is suitable, mention 2-3 alternatives, and use real-life examples."),
    ("user", "{question}")
])

chain = prompt | llm | StrOutputParser()

# -------------------
# Streamlit Chat
# -------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_question := st.chat_input("Ask about crops or soil..."):
    st.session_state.messages.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    enriched_question = (
        f"{user_question}\n\nSoil Data: {soil_data}\n"
        f"Weather Data: {weather_data}\n"
        f"ML Model Suggestion: {predicted_crop}\n"
        "Explain why this crop is suitable and give practical advice for a farmer."
    )

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        try:
            for chunk in chain.stream({"question": enriched_question}):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        except Exception as e:
            st.error(f"Error: {str(e)}")
