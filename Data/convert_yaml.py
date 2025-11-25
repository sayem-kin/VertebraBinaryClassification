import yaml
import re
from collections import Counter

# -----------------------------
# Function to process YAML file
# -----------------------------
def process_yaml(yaml_path):
    print(f"\nProcessing: {yaml_path}")

    # Load YAML
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    original_counts = Counter()
    new_counts = Counter()

    # Update all file names
    updated_list = []
    for item in data:
        filename = item
        
        # Extract label before .nii.gz
        match = re.search(r"_(\d)\.nii\.gz$", filename)
        if not match:
            print(f"⚠ Could not extract label from: {filename}")
            continue
        
        old_label = int(match.group(1))
        original_counts[old_label] += 1

        # Convert labels: 2→1, 3→1
        new_label = old_label if old_label in [0,1] else 1
        new_counts[new_label] += 1

        # Replace label in filename
        new_filename = re.sub(r"_(\d)\.nii\.gz$", f"_{new_label}.nii.gz", filename)
        updated_list.append(new_filename)

    # Save updated YAML
    with open(yaml_path, "w") as f:
        yaml.dump(updated_list, f)

    print("\n--- ORIGINAL LABEL COUNTS (0/1/2/3) ---")
    for k in sorted(original_counts):
        print(f"Label {k}: {original_counts[k]}")

    print("\n--- NEW BINARY LABEL COUNTS (0/1) ---")
    print(f"No fracture (0): {new_counts[0]}")
    print(f"Fracture (1): {new_counts[1]}")

    return original_counts, new_counts


# -----------------------------
# Run on both YAML files
# -----------------------------
train_yaml = r"D:\Binary Classification\Data\train_file_list.delx.yaml"
test_yaml = r"D:\Binary Classification\Data\test_file_list.delx.yaml"

process_yaml(train_yaml)
process_yaml(test_yaml)

print("\n✔ Conversion complete for both train and test YAML files.")