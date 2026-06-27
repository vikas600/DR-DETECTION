import os
import pandas as pd
import shutil

csv_path = "dataset/train.csv"
image_dir = "dataset/train_images"

base_out = "dataset/processed_multiclass/train"

label_map = {
    0: "No_DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

df = pd.read_csv(csv_path)

for _, row in df.iterrows():
    img = row["id_code"] + ".png"
    label = row["diagnosis"]

    src = os.path.join(image_dir, img)
    if not os.path.exists(src):
        continue

    dst = os.path.join(base_out, label_map[label], img)
    shutil.copy(src, dst)

print("✅ Multiclass dataset prepared")
