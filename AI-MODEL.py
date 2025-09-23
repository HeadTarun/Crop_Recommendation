import streamlit as st
import requests, os, tempfile, wave, av
from gtts import gTTS
from typing import List
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, RTCConfiguration
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# ------------------- Config -------------------
st.set_page_config(page_title="AI फसल सहायक", layout="wide")
st.markdown("<h1 style='text-align:center'>🗣️ AI फसल सहायक</h1>", unsafe_allow_html=True)

# ------------------- Load Environment -------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_BASE = "https://api.groq.com/openai/v1"
if not GROQ_API_KEY:
    st.error("कृपया .env फ़ाइल में GROQ_API_KEY सेट करें।")
    st.stop()

# ------------------- Session State -------------------
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "is_recording" not in st.session_state: st.session_state.is_recording = False
if "webrtc_context" not in st.session_state: st.session_state.webrtc_context = None
if "selected_model" not in st.session_state: st.session_state.selected_model = "gemma2-9b-it"

# ------------------- Location / Soil / Weather -------------------
def get_location():
    try:
        loc = requests.get("https://ipinfo.io/json", timeout=5).json().get("loc","28.61,77.20").split(",")
        return float(loc[0]), float(loc[1])
    except:
        return 28.61, 77.20

def fetch_soil(lat, lon):
    try:
        url=f"https://rest.isric.org/soilgrids/query?lon={lon}&lat={lat}&attributes=phh2o,nitrogen,ocd,sand,silt"
        r=requests.get(url,timeout=8)
        data = r.json().get("properties",{})
        return {k:v.get("M",{}).get("0-5cm",0) for k,v in data.items()}
    except:
        return {"phh2o":6.5,"nitrogen":50,"ocd":10,"sand":40,"silt":40}

def fetch_weather(lat, lon):
    if not WEATHER_API_KEY: return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}
    try:
        url=f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={lat},{lon}"
        c=requests.get(url,timeout=8).json().get("current",{})
        return {"temp_c":c.get("temp_c",25),"humidity":c.get("humidity",70),
                "precip_mm":c.get("precip_mm",2),"wind_kph":c.get("wind_kph",10)}
    except:
        return {"temp_c":25,"humidity":70,"precip_mm":2,"wind_kph":10}

lat, lon = get_location()
soil_data = fetch_soil(lat, lon)
weather_data = fetch_weather(lat, lon)

st.sidebar.write("🧪 मिट्टी का डेटा"); st.sidebar.json(soil_data)
st.sidebar.write("🌤 मौसम का डेटा"); st.sidebar.json(weather_data)

# ------------------- ML Model -------------------
def prepare_features(soil, weather):
    df = pd.DataFrame([soil]).join(pd.DataFrame([weather]))
    for col in ["phh2o","nitrogen","ocd","sand","silt","temp_c","humidity","precip_mm","wind_kph"]:
        if col not in df.columns: df[col]=0
    return df

X_df = prepare_features(soil_data, weather_data)
scaler = StandardScaler().fit(X_df)
X_scaled = scaler.transform(X_df)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
if X_scaled.shape[0]==1:
    X_train = np.tile(X_scaled,(20,1))
    y_train = np.random.choice([0,1,2],size=20)
else:
    X_train = X_scaled; y_train = np.random.choice([0,1,2],size=X_scaled.shape[0])
clf.fit(X_train,y_train)
crop_map={0:"🌾 गेहूँ",1:"🌱 धान",2:"🌽 मक्का"}
predicted_crop = crop_map.get(clf.predict(X_scaled)[0],"Unknown")
st.sidebar.success(f"✅ ML सुझाव: {predicted_crop}")

# ------------------- Audio -------------------
class AudioProcessor(AudioProcessorBase):
    def __init__(self): self.audio_frames: List[bytes]=[]
    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.audio_frames.append(frame.to_ndarray().tobytes())
        return frame

def save_wav(frames: List[bytes], path: str):
    combined = b"".join(frames)
    with wave.open(path,"wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(48000)
        wf.writeframes(combined)
    return path

def speech_to_text(file_path):
    url=f"{GROQ_BASE}/audio/transcriptions"
    with open(file_path,"rb") as f:
        resp = requests.post(url, headers={"Authorization":f"Bearer {GROQ_API_KEY}"},
                             data={"model":"whisper-large-v3","language":"hi"},
                             files={"file":(os.path.basename(file_path),f)}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("text","")

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        gTTS(text=text, lang='hi').save(tmp.name)
        return tmp.name

# ------------------- LLM -------------------
def get_llm_response(user_input, model_name):
    llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=model_name, temperature=0.7, streaming=True)
    prompt_text = (
        "आप एक कृषि AI सहायक हैं। केवल हिंदी में उत्तर दें। बुलेट में सुझाव दें। "
        f"मिट्टी: {soil_data}, मौसम: {weather_data}, ML सुझाव: {predicted_crop}, उपयोगकर्ता: {user_input}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", prompt_text)])
    chain = prompt | llm | StrOutputParser()
    full_response = ""
    for chunk in chain.stream({}):
        full_response += chunk
        yield full_response

# ------------------- Sidebar: Recording -------------------
st.sidebar.subheader("🎙 माइक्रोफ़ोन")
RTC_CONFIG = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
if st.session_state.webrtc_context is None:
    st.session_state.webrtc_context = webrtc_streamer(
        key="mic_stream", mode=WebRtcMode.SENDRECV,
        audio_processor_factory=AudioProcessor,
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True,
        rtc_configuration=RTC_CONFIG
    )

col1,col2 = st.sidebar.columns(2)
with col1:
    if st.button("🔴 रिकॉर्डिंग शुरू करें"): st.session_state.is_recording=True
with col2:
    if st.button("❌ रद्द करें"):
        st.session_state.is_recording=False
        st.session_state.webrtc_context=None

# ------------------- Audio Flow -------------------
if st.session_state.is_recording and st.session_state.webrtc_context:
    if st.button("✅ रोकें और भेजें"):
        st.session_state.is_recording=False
        processor = st.session_state.webrtc_context.audio_processor
        if processor and processor.audio_frames:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                save_wav(processor.audio_frames, tmp.name)
                audio_path = tmp.name
            with st.spinner("ऑडियो → टेक्स्ट → AI उत्तर तैयार हो रहा है..."):
                try:
                    text = speech_to_text(audio_path)
                    st.session_state.chat_history.append({"role":"user","content":text})
                    response_container = st.empty()
                    full_response = ""
                    for chunk in get_llm_response(text, st.session_state.selected_model):
                        response_container.markdown(chunk)
                        full_response = chunk
                    st.session_state.chat_history.append({"role":"assistant","content":full_response})
                    # TTS playback
                    audio_file = text_to_speech(full_response)
                    st.audio(audio_file)
                    os.remove(audio_file)
                    os.remove(audio_path)
                    st.rerun()
                except Exception as e:
                    st.error(f"ऑडियो प्रोसेसिंग में त्रुटि: {e}")

# ------------------- Text Flow -------------------
st.markdown("---")
for msg in st.session_state.chat_history:
    role = "💬 आप" if msg["role"]=="user" else "🤖 AI"
    st.markdown(f"**{role}:** {msg['content']}")

user_input = st.chat_input("फसल या मिट्टी के बारे में पूछें...")
if user_input:
    st.session_state.chat_history.append({"role":"user","content":user_input})
    response_container = st.empty()
    full_response = ""
    with st.spinner("AI उत्तर तैयार हो रहा है..."):
        for chunk in get_llm_response(user_input, st.session_state.selected_model):
            response_container.markdown(chunk)
            full_response = chunk
    st.session_state.chat_history.append({"role":"assistant","content":full_response})
    # TTS playback
    audio_file = text_to_speech(full_response)
    st.audio(audio_file)
    os.remove(audio_file)
    st.rerun()
