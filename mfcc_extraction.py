import os
import numpy as np
import librosa

# ============================
# CONFIG
# ============================

DATASET_DIR = "split_dataset"
TARGET_SR = 16000
MFCC_COUNT = 13

OUTPUT_DIR = "features"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3-class mapping
label_map = {
    "real": 0,
    "scam": 1,
    "ai_scam": 2
}


def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=MFCC_COUNT
        )

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var  = np.var(mfcc, axis=1)

        return np.concatenate([mfcc_mean, mfcc_var])

    except Exception as e:
        print("Error:", file_path, e)
        return None


def process_dataset():

    X = []
    y = []

    for split in ["train", "test"]:

        split_path = os.path.join(DATASET_DIR, split)

        for label_name in ["real", "scam", "ai_scam"]:

            label_path = os.path.join(split_path, label_name)

            if not os.path.exists(label_path):
                continue

            for file in os.listdir(label_path):

                if file.lower().endswith(".wav"):

                    file_path = os.path.join(label_path, file)

                    features = extract_features(file_path)

                    if features is not None:
                        X.append(features)
                        y.append(label_map[label_name])

    return np.array(X), np.array(y)


def main():

    print("Extracting MFCC features...")

    X, y = process_dataset()

    np.save(os.path.join(OUTPUT_DIR, "x.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\nMFCC extraction complete")
    print("Shape:", X.shape)


if __name__ == "__main__":
    main()