import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# ==============================
# LOAD DATA
# ==============================

# Audio features
X_audio = np.load("features/x.npy")
y = np.load("features/y.npy")

# Transcripts
df = pd.read_csv("transcripts.csv")
texts = df["transcript"].astype(str).values


# ==============================
# SAFETY CHECK (IMPORTANT)
# ==============================

assert len(X_audio) == len(texts), "Mismatch between audio features and transcripts"


# ==============================
# TRAIN TEST SPLIT
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
# AUDIO NORMALIZATION
# ==============================

scaler = StandardScaler()

X_train_audio = scaler.fit_transform(X_train_audio)
X_test_audio  = scaler.transform(X_test_audio)


# ==============================
# AUDIO MODEL
# ==============================

audio_model = RandomForestClassifier(n_estimators=150, random_state=42)

audio_model.fit(X_train_audio, y_train)

audio_pred = audio_model.predict(X_test_audio)


# ==============================
# TEXT MODEL
# ==============================

vectorizer = TfidfVectorizer(max_features=3000)

X_train_text = vectorizer.fit_transform(train_text)
X_test_text  = vectorizer.transform(test_text)

text_model = LogisticRegression(max_iter=1000)

text_model.fit(X_train_text, y_train)

text_pred = text_model.predict(X_test_text)


# ==============================
# FUSION LOGIC (3-CLASS SAFE)
# ==============================

final_pred = []

for a, t in zip(audio_pred, text_pred):
    if a == t:
        final_pred.append(a)
    else:
        final_pred.append(t)   # fallback

final_pred = np.array(final_pred)


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