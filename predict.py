import numpy as np
import joblib
import tempfile
import os

from mfcc_extraction import extract_features
from speech_to_text import transcribe_file as transcribe_audio

# -------------------------
# LOAD MODELS
# -------------------------
audio_model = joblib.load("models/audio_model.pkl")
text_model = joblib.load("models/text_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")
scaler = joblib.load("models/scaler.pkl")


def predict(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    try:
        # -------------------------
        # SPEECH TO TEXT
        # -------------------------
        transcript = transcribe_audio(temp_path)
        if not transcript:
            transcript = ""

        # -------------------------
        # FEATURES
        # -------------------------
        audio_features = extract_features(temp_path)
        audio_features = scaler.transform([audio_features])

        text_vector = vectorizer.transform([transcript])

        # -------------------------
        # MODEL PROBABILITIES
        # -------------------------
        audio_prob = audio_model.predict_proba(audio_features)[0]
        text_prob = text_model.predict_proba(text_vector)[0]

        # -------------------------
        # FUSION (text slightly stronger)
        # -------------------------
        final_prob = 0.3 * audio_prob + 0.7 * text_prob

        scam_score = final_prob[1]
        real_score = final_prob[0]

        # -------------------------
        # STRONGER SEPARATION LOGIC
        # -------------------------
        confidence = scam_score - real_score  # -1 to +1

        # Push neutral cases downward
        if abs(confidence) < 0.15:
            confidence -= 0.2

        score = int((confidence + 1) * 50)

        # -------------------------
        # KEYWORD BOOST (controlled)
        # -------------------------
        text = transcript.lower()

        scam_keywords = [
            "otp", "bank", "verify", "urgent",
            "account", "suspend", "fraud", "pin"
        ]

        boost = sum(1 for word in scam_keywords if word in text)

        if boost >= 2:
            score += 15
        elif boost == 1:
            score += 5

        # -------------------------
        # NORMAL SPEECH PENALTY
        # -------------------------
        normal_keywords = ["hello", "thank you", "okay", "meeting"]

        if any(word in text for word in normal_keywords) and boost == 0:
            score -= 15

        # Clamp score
        score = max(0, min(score, 100))

        # -------------------------
        # VERDICT
        # -------------------------
        if score > 75:
            verdict = "Likely Scam"
        elif score > 60:
            verdict = "Suspicious"
        elif score > 35:
            verdict = "Caution"
        else:
            verdict = "Safe"

        return {
            "score": score,
            "verdict": verdict,
            "transcript": transcript
        }

    finally:
        os.remove(temp_path)