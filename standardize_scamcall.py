import os
import numpy as np
import librosa
import soundfile as sf

INPUT_DIR = "vishing-detection-system/standardized_scamcalls"
OUTPUT_DIR = "vishing-detection-system/standardized_scamcalls_segments"

TARGET_SR = 16000
SEGMENT_DURATION = 30
MFCC_COUNT = 13


def extract_features(audio, sr):

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=MFCC_COUNT
    )

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    features = np.concatenate([mfcc_mean, mfcc_var])

    return features


def standardize_audio(input_path):

    features_list = []

    try:
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        segment_length = TARGET_SR * SEGMENT_DURATION

        total_segments = len(audio) // segment_length + 1

        filename = os.path.splitext(os.path.basename(input_path))[0]

        for i in range(total_segments):

            start = i * segment_length
            end = start + segment_length

            segment = audio[start:end]

            if len(segment) < TARGET_SR:
                continue

            output_name = f"{filename}_seg{i}.wav"
            output_path = os.path.join(OUTPUT_DIR, output_name)

            sf.write(output_path, segment, TARGET_SR)

            features = extract_features(segment, TARGET_SR)

            features_list.append(features)

        print("Processed:", input_path)

    except Exception as e:
        print("Error:", input_path, e)

    return features_list


def main():

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = []

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".wav", ".mp3"))
    ]

    print("Total files:", len(files))

    for file in files:

        input_path = os.path.join(INPUT_DIR, file)

        features = standardize_audio(input_path)

        X.extend(features)

    X = np.array(X)

    np.save("x.npy", X)

    print("\nStandardization complete")
    print("Saved audio to:", OUTPUT_DIR)
    print("Feature shape:", X.shape)


if __name__ == "__main__":
    main()