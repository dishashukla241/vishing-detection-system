import numpy as np
import pandas as pd
import joblib
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# -------------------------
# LOAD AUDIO FEATURES
# -------------------------
X_audio = np.load("features/x.npy")
y = np.load("features/y.npy")

# -------------------------
# LOAD TEXT DATA
# -------------------------
text_df = pd.read_csv("transcripts.csv")
X_text_raw = text_df["transcript"].fillna("").astype(str)

# -------------------------
# ALIGNMENT CHECK
# -------------------------
assert X_audio.shape[0] == len(X_text_raw) == y.shape[0], "Data mismatch!"

# -------------------------
# TEXT FEATURES
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 3),
    stop_words="english"
)

X_text = vectorizer.fit_transform(X_text_raw)

# -------------------------
# AUDIO SCALING
# -------------------------
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# -------------------------
# SPLIT
# -------------------------
Xa_train, Xa_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
    X_audio_scaled,
    X_text,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# MODELS
# -------------------------
audio_model = GradientBoostingClassifier()
text_model = GradientBoostingClassifier()

audio_model.fit(Xa_train, y_train)
text_model.fit(Xt_train, y_train)

# -------------------------
# PREDICTIONS
# -------------------------
audio_pred = audio_model.predict(Xa_test)
text_pred = text_model.predict(Xt_test)

# -------------------------
# FUSION (IMPORTANT)
# -------------------------
audio_probs = audio_model.predict_proba(Xa_test)
text_probs = text_model.predict_proba(Xt_test)

final_probs = 0.5 * audio_probs + 0.5 * text_probs
final_pred = np.argmax(final_probs, axis=1)

# -------------------------
# METRICS
# -------------------------
metrics = {
    "audio": {
        "accuracy": accuracy_score(y_test, audio_pred),
        "precision": precision_score(y_test, audio_pred, average="weighted"),
        "recall": recall_score(y_test, audio_pred, average="weighted"),
        "f1": f1_score(y_test, audio_pred, average="weighted")
    },
    "text": {
        "accuracy": accuracy_score(y_test, text_pred),
        "precision": precision_score(y_test, text_pred, average="weighted"),
        "recall": recall_score(y_test, text_pred, average="weighted"),
        "f1": f1_score(y_test, text_pred, average="weighted")
    },
    "fusion": {
        "accuracy": accuracy_score(y_test, final_pred),
        "precision": precision_score(y_test, final_pred, average="weighted"),
        "recall": recall_score(y_test, final_pred, average="weighted"),
        "f1": f1_score(y_test, final_pred, average="weighted")
    }
}

# -------------------------
# SAVE EVERYTHING
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(audio_model, "models/audio_model.pkl")
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# SAVE METRICS
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Models + metrics saved successfully!")