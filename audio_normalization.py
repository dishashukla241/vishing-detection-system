import os
import librosa
import soundfile as sf
import numpy as np

# =============================
# CONFIG
# =============================

DATASET_FOLDERS = [
    "output_split_aiscamcalls",
    "standardized_realcalls",
    "standardized_scamcalls_segments"
]

OUTPUT_BASE = "normalized_datasets"

TARGET_SR = 16000


def amplitude_normalize(audio):

    max_val = np.max(np.abs(audio))

    if max_val == 0:
        return audio

    return audio / max_val


def process_file(input_path, output_path):

    try:
        audio, sr = librosa.load(input_path, sr=TARGET_SR, mono=True)

        normalized_audio = amplitude_normalize(audio)

        sf.write(output_path, normalized_audio, TARGET_SR)

    except Exception as e:
        print("Error:", input_path, e)


def process_dataset(dataset_path):

    dataset_name = os.path.basename(dataset_path)

    output_dataset = os.path.join(OUTPUT_BASE, dataset_name)

    os.makedirs(output_dataset, exist_ok=True)

    for root, dirs, files in os.walk(dataset_path):

        rel_path = os.path.relpath(root, dataset_path)

        out_dir = os.path.join(output_dataset, rel_path)

        os.makedirs(out_dir, exist_ok=True)

        for file in files:

            if file.lower().endswith((".wav", ".mp3")):

                input_path = os.path.join(root, file)

                new_name = os.path.splitext(file)[0] + ".wav"

                output_path = os.path.join(out_dir, new_name)

                process_file(input_path, output_path)


def main():

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    for dataset in DATASET_FOLDERS:

        print("Processing:", dataset)

        process_dataset(dataset)

    print("\nAmplitude normalization complete")


if __name__ == "__main__":
    main()