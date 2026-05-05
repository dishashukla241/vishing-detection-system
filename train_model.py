import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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
# 🔍 ALIGNMENT CHECK (VERY IMPORTANT)
# -------------------------
print("Audio samples:", X_audio.shape[0])
print("Text samples:", len(X_text_raw))
print("Labels:", y.shape[0])

assert X_audio.shape[0] == len(X_text_raw) == y.shape[0], "❌ Data mismatch!"

# Show sample mapping
print("\nSample check:")
for i in range(3):
    print(f"Label: {y[i]} | Text: {X_text_raw[i][:50]}")

# -------------------------
# TEXT PROCESSING (IMPROVED)
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
# SINGLE SPLIT (CORRECT)
# -------------------------
Xa_train, Xa_test, Xt_train, Xt_test, y_train, y_test = train_test_split(
    X_audio_scaled,
    X_text,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# BETTER MODELS
# -------------------------
audio_model = GradientBoostingClassifier()
text_model = GradientBoostingClassifier()

audio_model.fit(Xa_train, y_train)
text_model.fit(Xt_train, y_train)

# -------------------------
# EVALUATION
# -------------------------
audio_pred = audio_model.predict(Xa_test)
text_pred = text_model.predict(Xt_test)

print("\n📊 Model Performance:")
print("Audio Accuracy:", accuracy_score(y_test, audio_pred))
print("Text Accuracy:", accuracy_score(y_test, text_pred))

# -------------------------
# SAVE MODELS
# -------------------------
os.makedirs("models", exist_ok=True)

joblib.dump(audio_model, "models/audio_model.pkl")
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Models trained and saved successfully!")