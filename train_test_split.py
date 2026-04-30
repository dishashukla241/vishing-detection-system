import os
import shutil
from sklearn.model_selection import train_test_split
from collections import defaultdict, Counter

# =============================
# INPUT DIRECTORIES
# =============================
DATA_DIRS = {
    "normalized_datasets/standardized_realcalls": "real",
    "normalized_datasets/standardized_scamcalls_segments": "scam",
    "normalized_datasets/output_split_aiscamcalls": "ai_scam"
}

OUTPUT_DIR = "split_dataset"

# =============================
# CREATE OUTPUT STRUCTURE
# =============================
for split in ["train", "test"]:
    for label in ["real", "scam", "ai_scam"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, label), exist_ok=True)

# =============================
# GROUPING FUNCTION (FIXED)
# =============================
def get_group_key(file, label):
    parts = file.split("_")

    if label == "real":
        # Example: F2308F2308_USA_USA_001_seg0.wav
        # Group by call ID (001)
        return label + "_" + "_".join(parts[:4])

    elif label == "scam":
        # Example: scam_10_seg0.wav
        return label + "_" + parts[1]

    elif label == "ai_scam":
        # Example: 1002976_normalized_seg0.wav
        return label + "_" + parts[0]

# =============================
# STEP 1: GROUP FILES
# =============================
groups = defaultdict(list)

for folder, label in DATA_DIRS.items():
    for file in os.listdir(folder):
        if file.lower().endswith(".wav"):
            key = get_group_key(file, label)
            groups[key].append((folder, file, label))

# =============================
# DEBUG: CHECK GROUP COUNTS
# =============================
group_keys = list(groups.keys())
labels = [key.split("_")[0] for key in group_keys]

print("Before Split:", Counter(labels))

real_groups = [k for k in group_keys if k.startswith("real")]
print("Total REAL groups:", len(real_groups))

# =============================
# STEP 2: STRATIFIED SPLIT
# =============================
train_keys, test_keys = train_test_split(
    group_keys,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

# Debug after split
train_labels = [key.split("_")[0] for key in train_keys]
test_labels = [key.split("_")[0] for key in test_keys]

print("Train:", Counter(train_labels))
print("Test:", Counter(test_labels))

# =============================
# STEP 3: COPY FILES
# =============================
def copy_files(keys, split):
    for key in keys:
        for folder, file, label in groups[key]:
            src = os.path.join(folder, file)
            dst = os.path.join(OUTPUT_DIR, split, label, file)
            shutil.copy(src, dst)

copy_files(train_keys, "train")
copy_files(test_keys, "test")

print("✅ Dataset created successfully with ALL 3 classes!")