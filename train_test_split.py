import os
import shutil
from sklearn.model_selection import train_test_split

# INPUT DIRECTORIES
DATA_DIRS = {
    "output_split_aiscamcalls": "ai_scam",
    "standardized_realcalls": "real",
    "standardized_scamcalls_segments": "scam"
}

OUTPUT_DIR = "split_dataset"

# Create output structure
for split in ["train", "test"]:
    for label in ["real", "scam", "ai_scam"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# Step 1: Group segments
groups = {}

for folder, label in DATA_DIRS.items():
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            base = "_".join(file.split("_")[:2])

            if base not in groups:
                groups[base] = []

            groups[base].append((folder, file, label))

# Step 2: Split groups
group_keys = list(groups.keys())

train_keys, test_keys = train_test_split(
    group_keys, test_size=0.2, random_state=42
)

# Step 3: Copy files
def copy_files(keys, split):
    for key in keys:
        for folder, file, label in groups[key]:

            src = os.path.join(folder, file)
            dst = os.path.join(OUTPUT_DIR, split, label, file)

            shutil.copy(src, dst)

# Copy data
copy_files(train_keys, "train")
copy_files(test_keys, "test")

print("Dataset created with 3 classes!")