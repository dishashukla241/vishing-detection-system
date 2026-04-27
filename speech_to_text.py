import os
import whisper
import pandas as pd

INPUT_DIR = "split_dataset"
OUTPUT_FILE = "transcripts.csv"

MODEL_SIZE = "base"

model = whisper.load_model(MODEL_SIZE)


def transcribe_file(file_path):
    try:
        result = model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print("Error:", file_path, e)
        return ""


def process_dataset():
    rows = []

    for split in ["train", "test"]:
        split_path = os.path.join(INPUT_DIR, split)

        if not os.path.exists(split_path):
            continue

        for label in os.listdir(split_path):
            label_path = os.path.join(split_path, label)

            for file in os.listdir(label_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(label_path, file)

                    print("Processing:", file)

                    text = transcribe_file(file_path)

                    rows.append({
                        "file": file,
                        "split": split,
                        "label": label,
                        "transcript": text
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nTranscription complete")
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    process_dataset()