import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import warnings
import numpy as np
from lime.lime_text import LimeTextExplainer
import os
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import io

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["STREAMLIT_WATCHER_IGNORE"] = "torch"

# Streamlit settings
st.set_page_config(page_title="Lie Detector", page_icon="🕵️", layout="centered")

MODEL_PATH = "deception-bert-model"

@st.cache_resource
def load_model_and_tokenizer():
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        model.to(torch.device("cpu"))
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error("❌ Failed to load the model.")
        st.exception(e)
        return None, None

# Load model
with st.spinner("Loading model..."):
    tokenizer, model = load_model_and_tokenizer()

# Page Title
st.title("🕵️ Lie Detector using BERT")
st.markdown("Enter a statement or record your voice to check if it's **truth** or **lie**.")

# Record from mic
audio = mic_recorder(
    start_prompt="🎙️ Start Recording",
    stop_prompt="⏹️ Stop",
    just_once=True,
    key="recorder"
)

# Initialize session state
if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

# Handle audio input
if audio:
    st.audio(audio["bytes"], format="audio/wav")
    
    st.write("📌 Debug: Raw audio buffer size =", len(audio["bytes"]))

    recognizer = sr.Recognizer()
    try:
        audio_file = sr.AudioFile(io.BytesIO(audio["bytes"]))
        with audio_file as source:
            st.write("🎧 Reading audio for transcription...")
            recorded_audio = recognizer.record(source)
        
        st.write("🔍 Trying Google Speech Recognition...")
        transcribed = recognizer.recognize_google(recorded_audio)
        st.session_state.transcribed_text = transcribed
        st.success(f"📝 Transcription: {transcribed}")

    except sr.UnknownValueError:
        st.error("❌ Could not understand audio. Try speaking clearly or closer to mic.")
        st.session_state.transcribed_text = ""
    except sr.RequestError as e:
        st.error(f"❌ Google API error: {e}")
        st.session_state.transcribed_text = ""
    except Exception as e:
        st.error("❌ Unexpected error while transcribing audio.")
        st.exception(e)
        st.session_state.transcribed_text = ""

# Text area
text_input = st.text_area("✍️ Statement:", value=st.session_state.transcribed_text, height=150)

# Labels
labels = ["Truth", "Lie"]

def predict_proba(texts):
    tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**tokens)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()
    return probs

# Analyze button
if st.button("🔍 Analyze"):
    if model is None:
        st.error("Model not loaded.")
    elif not text_input.strip():
        st.warning("Please enter or record a statement.")
    else:
        with torch.no_grad():
            inputs = tokenizer(text_input, return_tensors="pt", truncation=True, padding=True)
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0][prediction].item()
            label = labels[prediction]

            if label == "Truth":
                st.success(f"🟢 Prediction: **{label}** (Confidence: {confidence:.2%})")
            else:
                st.error(f"🔴 Prediction: **{label}** (Confidence: {confidence:.2%})")

        # Explain with LIME
        explainer = LimeTextExplainer(class_names=labels)
        with st.spinner("Generating explanation..."):
            explanation = explainer.explain_instance(
                text_input,
                predict_proba,
                num_features=6,
                num_samples=100
            )

        st.markdown("### 🧠 Influential Words:")
        for word, weight in explanation.as_list():
            color = "green" if weight > 0 else "red"
            st.markdown(f"<span style='color:{color}'>{word}</span>: **{weight:.3f}**", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Made with 🤖 BERT, 🎙️ Speech Recognition & 🧠 LIME | Powered by OpenAI + Streamlit")
