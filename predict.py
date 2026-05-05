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
        # FEATURE EXTRACTION
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
        # FUSION
        # -------------------------
        final_prob = 0.5 * audio_prob + 0.5 * text_prob

        # -------------------------
        # MULTI-CLASS SCORING (FINAL CALIBRATION)
        # -------------------------
        # index: 0 = real, 1 = scam, 2 = ai_scam
        real_prob = final_prob[0]

        # Boost AI scam slightly
        scam_prob = final_prob[1] + 1.5 * final_prob[2]
        scam_prob = min(scam_prob, 1.0)

        # Apply curve + downward shift
        score = int((scam_prob ** 1.4) * 100) - 10

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
            score += 10
        elif boost == 1:
            score += 4

        # -------------------------
        # NORMAL SPEECH PENALTY
        # -------------------------
        normal_keywords = ["hello", "thank you", "okay", "meeting"]

        if any(word in text for word in normal_keywords) and boost == 0:
            score -= 10

        # -------------------------
        # CLAMP SCORE
        # -------------------------
        score = max(0, min(score, 100))

        # -------------------------
        # FINAL VERDICT
        # -------------------------
        if score > 75:
            verdict = "Likely Scam"
        elif score > 55:
            verdict = "Suspicious"
        elif score > 30:
            verdict = "Caution"
        else:
            verdict = "Safe"

        return {
            "score": score,
            "verdict": verdict,
            "transcript": transcript,
            "probabilities": {
                "real": float(real_prob),
                "scam": float(final_prob[1]),
                "ai_scam": float(final_prob[2])
            }
        }

    finally:
        os.remove(temp_path)