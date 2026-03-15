import os
import whisper
import pandas as pd


INPUT_DIR = "split_dataset"
OUTPUT_FILE = "transcripts.csv"
MODEL_SIZE = "tiny"

# ==============================
# LOAD MODEL
# ==============================

model = whisper.load_model(MODEL_SIZE)

rows = []

for split in ["train", "test"]:

    split_path = os.path.join(INPUT_DIR, split)

    for label in os.listdir(split_path):

        label_path = os.path.join(split_path, label)

        for file in os.listdir(label_path):

            if file.endswith(".wav"):

                file_path = os.path.join(label_path, file)

                print("Processing:", file)

                result = model.transcribe(file_path, fp16=False)

                rows.append({
                    "file": file,
                    "split": split,
                    "label": label,
                    "transcript": result["text"]
                })

df = pd.DataFrame(rows)
df.to_csv(OUTPUT_FILE, index=False)

print("\nTranscription complete")
print("Saved to:", OUTPUT_FILE)