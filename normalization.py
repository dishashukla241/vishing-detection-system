import librosa
import numpy as np
import soundfile as sf
import os

# input folders
input_folders = [
    "standardized_realcalls",
    "split_scamcalls",
    "standardized_scamcalls"
]

# main output folder
output_base = "normalized_audio"

os.makedirs(output_base, exist_ok=True)

for folder in input_folders:

    output_folder = os.path.join(output_base, folder)
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder):

        if file.endswith(".wav"):

            file_path = os.path.join(folder, file)

            audio, sr = librosa.load(file_path, sr=None)

            normalized_audio = audio / np.max(np.abs(audio))

            output_path = os.path.join(output_folder, file)

            sf.write(output_path, normalized_audio, sr)

            print("Processed:", file)

print("All audio files normalized successfully")