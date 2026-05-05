import numpy as np
import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# LOAD DATA
# ==============================

X_audio = np.load("features/x.npy")
y = np.load("features/y.npy")

df = pd.read_csv("transcripts.csv")
df["transcript"] = df["transcript"].fillna("").astype(str)

# remove empty / short text
valid_idx = df["transcript"].str.strip().str.len() > 10

X_audio = X_audio[valid_idx.values]
y = y[valid_idx.values]
texts = df.loc[valid_idx, "transcript"].reset_index(drop=True)

# ==============================
# SPLIT
# ==============================

X_train_audio, X_test_audio, y_train, y_test, train_text, test_text = train_test_split(
    X_audio,
    y,
    texts,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==============================
# AUDIO PROCESSING
# ==============================

scaler = StandardScaler()
X_train_audio = scaler.fit_transform(X_train_audio)
X_test_audio  = scaler.transform(X_test_audio)

# ==============================
# TEXT PROCESSING
# ==============================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,3),
    stop_words="english"
)

X_train_text = vectorizer.fit_transform(train_text)
X_test_text  = vectorizer.transform(test_text)

# ==============================
# MODELS (FINAL)
# ==============================

audio_model = GradientBoostingClassifier()
text_model  = GradientBoostingClassifier()

audio_model.fit(X_train_audio, y_train)
text_model.fit(X_train_text, y_train)

# ==============================
# PREDICTIONS
# ==============================

audio_pred = audio_model.predict(X_test_audio)
text_pred  = text_model.predict(X_test_text)

audio_probs = audio_model.predict_proba(X_test_audio)
text_probs  = text_model.predict_proba(X_test_text)

# ==============================
# FUSION
# ==============================

final_probs = 0.4 * audio_probs + 0.6 * text_probs
final_pred = np.argmax(final_probs, axis=1)

# ==============================
# EVALUATION
# ==============================

print("\n===== AUDIO MODEL =====")
print("Accuracy:", accuracy_score(y_test, audio_pred))
print(classification_report(y_test, audio_pred))

print("\n===== TEXT MODEL =====")
print("Accuracy:", accuracy_score(y_test, text_pred))
print(classification_report(y_test, text_pred))

print("\n===== FINAL FUSION MODEL =====")
print("Accuracy:", accuracy_score(y_test, final_pred))
print(classification_report(y_test, final_pred))

# ==============================
# SAVE MODELS
# ==============================

os.makedirs("models", exist_ok=True)

joblib.dump(audio_model, "models/audio_model.pkl")
joblib.dump(text_model, "models/text_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n✅ Final models saved!")