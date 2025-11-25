import os
import re
from collections import Counter

# Path to your folder
folder = r"D:\Binary Classification\Data\img"

# Regex to extract the label before '.nii'
pattern = re.compile(r"_(\d)\.nii\.gz$")

counts = Counter()

for filename in os.listdir(folder):
    match = pattern.search(filename)
    if not match:
        continue

    old_label = int(match.group(1))

    # Convert multiclass → binary
    if old_label in [2, 3]:
        new_label = 1
    else:
        new_label = old_label

    # Count new label
    counts[new_label] += 1

    # Create new filename
    new_filename = re.sub(r"_(\d)\.nii\.gz$", f"_{new_label}.nii.gz", filename)

    # Rename file only if different
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_filename)

    if old_path != new_path:
        os.rename(old_path, new_path)

# Print final counts
print("Final counts after binary conversion:")
print(f"No fracture (0): {counts[0]}")
print(f"Fracture (1): {counts[1]}")