import io
import requests
import nltk
import streamlit as st
import speech_recognition as sr

from PIL import Image
from gtts import gTTS
from nltk.tokenize import word_tokenize

# --------------------------
# CONFIG
# --------------------------

HF_TOKEN = "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"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

nltk.download("punkt")

# --------------------------
# FUNCTIONS
# --------------------------

def split_into_meaningful_words(text):
    words = word_tokenize(text)

    meaningful_words = [
        word for word in words
        if word.isalnum()
    ]

    return ", ".join(meaningful_words)


def chatbot_response(prompt):

    API_URL = (
        "https://api-inference.huggingface.co/models/google/flan-t5-large"
    )

    payload = {
        "inputs": (
            "You are a health assistant. "
            "Give general wellness guidance only. "
            "Do not diagnose diseases.\n\n"
            + prompt
        )
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    result = response.json()

    try:
        return result[0]["generated_text"]

    except Exception:
        return str(result)


def summarize_text(text):

    API_URL = (
        "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    )

    payload = {
        "inputs": text,
        "options": {
            "wait_for_model": True
        }
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    return response.json()


def generate_image(prompt):

    API_URL = (
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
    )

    payload = {
        "inputs": prompt
    }

    response = requests.post(
        API_URL,
        headers=headers,
        json=payload
    )

    return response.content


# --------------------------
# PAGE
# --------------------------

st.set_page_config(
    page_title="DiagnoAI",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 DiagnoAI")
st.subheader("AI Health Assistant")

# --------------------------
# CHAT HISTORY
# --------------------------

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",
            "content": "Upload a WAV file and I will analyze it."
        }

    ]

for msg in st.session_state.messages:

    st.chat_message(
        msg["role"]
    ).write(
        msg["content"]
    )

# --------------------------
# AUDIO UPLOAD
# --------------------------

audio_file = st.file_uploader(
    "Upload WAV Audio",
    type=["wav"]
)

recognizer = sr.Recognizer()

if audio_file:

    st.audio(audio_file)

    with open("temp.wav", "wb") as f:

        f.write(
            audio_file.read()
        )

    # ----------------------
    # SPEECH TO TEXT
    # ----------------------

    with st.spinner(
        "Transcribing..."
    ):

        with sr.AudioFile(
            "temp.wav"
        ) as source:

            audio = recognizer.record(
                source
            )

        transcribed_text = (
            recognizer.recognize_google(
                audio
            )
        )

    st.chat_message(
        "user"
    ).write(
        transcribed_text
    )

    # ----------------------
    # AI RESPONSE
    # ----------------------

    with st.spinner(
        "Generating response..."
    ):

        ai_response = chatbot_response(
            transcribed_text
        )

    st.chat_message(
        "assistant"
    ).write(
        ai_response
    )

    # ----------------------
    # TEXT TO SPEECH
    # ----------------------

    with st.spinner(
        "Generating voice..."
    ):

        speech = gTTS(
            text=ai_response,
            lang="en"
        )

        speech.save(
            "response.mp3"
        )

    st.audio(
        "response.mp3"
    )

    # ----------------------
    # SUMMARIZE
    # ----------------------

    with st.spinner(
        "Summarizing..."
    ):

        summary = summarize_text(
            ai_response
        )

        prompt_words = (
            split_into_meaningful_words(
                str(summary)
            )
        )

    # ----------------------
    # IMAGE GENERATION
    # ----------------------

    with st.spinner(
        "Generating image..."
    ):

        image_prompt = (
            prompt_words
            + ", healthy lifestyle, doctor, exercise, fruits, vegetables"
        )

        image_bytes = generate_image(
            image_prompt
        )

        image = Image.open(
            io.BytesIO(
                image_bytes
            )
        )

    st.image(
        image,
        caption="AI Generated Health Image",
        use_container_width=True
    )

    st.success(
        "Analysis Completed"
    )
