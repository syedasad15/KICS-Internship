
import streamlit as st
import numpy as np
import joblib
import warnings
import os
import librosa
import soundfile as sf
import io
from streamlit_mic_recorder import mic_recorder
from sklearn.preprocessing import StandardScaler

# Ignore warnings and logs
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Page setup
st.set_page_config(page_title="Lie Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ Lie Detector using Audio Signals")
st.markdown("🎙️ Record your voice or upload a `.wav` file and click **Analyze** to detect if it's **truth** or **lie** based on acoustic features.")

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    return model

model = load_model()
labels = ["Truth", "Lie"]

# Set up session state
if "audio_bytes" not in st.session_state:
    st.session_state.audio_bytes = None

# Audio input section
st.subheader("🎙️ Voice Input")

# Microphone recorder
audio = mic_recorder(start_prompt="🎙️ Start Recording", stop_prompt="⏹️ Stop", just_once=True, key="mic")

if audio:
    st.session_state.audio_bytes = audio["bytes"]
    st.audio(audio["bytes"], format="audio/wav")
    st.success("✅ Audio recorded successfully.")

# File uploader
uploaded_file = st.file_uploader("📂 Or upload a .wav file", type=["wav"])
if uploaded_file is not None:
    st.session_state.audio_bytes = uploaded_file.read()
    st.audio(st.session_state.audio_bytes, format="audio/wav")
    st.success("✅ File uploaded successfully.")

# Feature extraction
def extract_features(audio_bytes, sr_target=16000):
    y, sr = sf.read(io.BytesIO(audio_bytes))
    if y.ndim > 1:  # stereo to mono
        y = np.mean(y, axis=1)
    if sr != sr_target:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_target)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_target, n_mfcc=30)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean.reshape(1, -1)

# Analyze button
if st.button("🔍 Analyze"):
    if not st.session_state.audio_bytes:
        st.warning("Please record or upload an audio file.")
    else:
        try:
            st.info("🔄 Extracting features...")
            features = extract_features(st.session_state.audio_bytes)
            st.success("✅ Features extracted.")

            st.info("🧠 Running prediction with custom threshold...")
            proba = model.predict_proba(features)[0]
            lie_prob = proba[1]

            if lie_prob > 0.7:
                label = "Lie"
                st.error(f"🔴 Prediction: **{label}** (Confidence: {lie_prob:.2%})")
            else:
                label = "Truth"
                st.success(f"🟢 Prediction: **{label}** (Confidence: {1 - lie_prob:.2%})")

        except Exception as e:
            st.error("❌ Failed to process audio or run prediction.")
            st.exception(e)

# Footer
st.markdown("---")
st.caption("Developed by NCAI ICRL Lab, KICS UET Lahore")
st.caption("Made with 🔊 Audio Features, 🧠 Voting Classifier & Streamlit")
