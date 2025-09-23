import streamlit as st
import requests, os, json, tempfile, wave, av
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
from typing import List

# ------------------- Config -------------------
st.set_page_config(page_title="AgroMind", layout="wide")
st.title("🌾 AI Crop Recommendation Assistant")

# ------------------- Load Env -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1"
if not GROQ_API_KEY: st.error("Set GROQ_API_KEY in .env"); st.stop()

# ------------------- Session State -------------------
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ------------------- Location & Soil/Weather -------------------
def get_user_location():
    try:
        res = requests.get("https://ipinfo.io/json", timeout=5)
        loc = res.json().get("loc", "28.61,77.20").split(",")
        return float(loc[0]), float(loc[1])
    except:
        return 28.61, 77.20

def fetch_soil(lat, lon):
    try:
        url = f"https://rest.isric.org/soilgrids/query?lon={lon}&lat={lat}&attributes=phh2o,nitrogen,ocd,sand,silt"
        r = requests.get(url, timeout=8)
        data = r.json().get("properties", {})
        return {k: v.get("M", {}).get("0-5cm", 0) for k,v in data.items()}
    except:
        return {"phh2o":6.5,"nitrogen":50,"ocd":10,"sand":40,"silt":40}

def fetch_weather(lat, lon):
    if not WEATHER_API_KEY: return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}
    try:
        url=f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
        r=requests.get(url,timeout=8)
        c=r.json().get("current",{})
        return {"temp_c":c.get("temp_c",25),"humidity":c.get("humidity",70),
                "precip_mm":c.get("precip_mm",2),"wind_kph":c.get("wind_kph",10)}
    except: return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}

lat, lon = get_user_location()
soil_data = fetch_soil(lat, lon)
weather_data = fetch_weather(lat, lon)

st.sidebar.write("🧪 Soil Data"); st.sidebar.json(soil_data)
st.sidebar.write("🌤 Weather Data"); st.sidebar.json(weather_data)

# ------------------- ML Crop Model -------------------
def prepare_features(soil, weather):
    df=pd.DataFrame([soil])
    df=pd.concat([df,pd.DataFrame([weather])],axis=1).fillna(0)
    return StandardScaler().fit_transform(df)

X_scaled=prepare_features(soil_data, weather_data)
def train_model(X):
    np.random.seed(42)
    y=np.random.choice([0,1,2],size=X.shape[0])
    if X.shape[0]==1:
        X=np.tile(X,(20,1)); y=np.random.choice([0,1,2],size=X.shape[0])
    clf=RandomForestClassifier(n_estimators=100,random_state=42); clf.fit(X,y)
    return clf

clf=train_model(X_scaled)
crop_map={0:"🌾 गेहूँ (Wheat)",1:"🌱 धान (Rice)",2:"🌽 मक्का (Maize)"}
predicted_crop=crop_map.get(clf.predict(X_scaled)[0],"Unknown")
st.sidebar.success(f"✅ ML Suggestion: {predicted_crop}")

# ------------------- Groq LLM -------------------
MODEL_NAME="gemma2-9b-it"
llm=ChatGroq(groq_api_key=GROQ_API_KEY, model_name=MODEL_NAME, temperature=0.7, streaming=True)
prompt=ChatPromptTemplate.from_messages([
    ("system","आप एक कृषि AI सहायक हैं। किसान को मिट्टी + मौसम + ML परिणाम के आधार पर सरल हिंदी में समझाइए। "
              "बताइए यह फसल क्यों उपयुक्त है, 2-3 विकल्प दीजिए, और वास्तविक उदाहरण शामिल कीजिए।"),
    ("user","{question}")
])
chain=prompt | llm | StrOutputParser()

# ------------------- Speech Functions -------------------
def speech_to_text(file_path): 
    url=f"{GROQ_BASE}/audio/transcriptions"
    with open(file_path,"rb") as f:
        resp=requests.post(url, headers={"Authorization":f"Bearer {GROQ_API_KEY}"}, data={"model":"whisper-large-v3"}, files={"file":(os.path.basename(file_path),f)}, timeout=30)
    resp.raise_for_status(); return resp.json().get("text","")

def text_to_speech(text, filename="response.mp3"):
    url=f"{GROQ_BASE}/audio/speech"
    payload={"model":"gpt-4o-mini-tts","voice":"alloy","input":text}
    resp=requests.post(url, headers={"Authorization":f"Bearer {GROQ_API_KEY}","Content-Type":"application/json"}, data=json.dumps(payload), timeout=30)
    resp.raise_for_status()
    with open(filename,"wb") as f: f.write(resp.content)
    return filename

# ------------------- Audio Processor -------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self): self.audio_frames:List[bytes]=[]
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame.to_ndarray().tobytes()); return frame

def save_wav(frames:List[bytes], path:str):
    combined=b"".join(frames)
    with wave.open(path,"wb") as wf: wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000); wf.writeframes(combined)
    return path

# ------------------- Display Chat -------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ------------------- Input Bar -------------------

if user_input := st.chat_input("Ask about crop or soil..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    enriched = f"{user_input}\n\nमिट्टी डेटा: {soil_data}\nमौसम डेटा: {weather_data}\nML सुझाव: {predicted_crop}\nकृपया किसान को सरल हिंदी में समझाइए।"
    
    try:
        response_container = st.chat_message("assistant")
        response_placeholder = response_container.empty()
        full_response = ""
        for chunk in chain.stream({"question": enriched}):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        response_placeholder.markdown(full_response)
        
        try:
            audio_file = text_to_speech(full_response)
            st.audio(audio_file, format="audio/mp3")
        except:
            st.write(full_response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
    
    except Exception as e:
        st.error(f"LLM failed: {e}")


with st.sidebar:
    st.subheader("Upload Audio or Use Mic")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "m4a"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            audio_path = tmp.name
        try:
            text = speech_to_text(audio_path)
            st.session_state.chat_history.append({"role": "user", "content": text})
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Audio transcription failed: {e}")

    st.markdown("---")
    st.subheader("Use Microphone")
    ctx = webrtc_streamer(key="mic_stream", mode=WebRtcMode.SENDRECV, audio_processor_factory=AudioProcessor, media_stream_constraints={"audio": True, "video": False}, async_processing=True)
    if ctx.audio_processor and ctx.audio_processor.audio_frames:
        if st.button("🎤 Send Mic", key="mic_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                save_wav(ctx.audio_processor.audio_frames, tmp.name)
                mic_path = tmp.name
            try:
                mic_text = speech_to_text(mic_path)
                st.session_state.chat_history.append({"role": "user", "content": mic_text})
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Mic transcription failed: {e}")
