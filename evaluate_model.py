import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

IMG_SIZE = 224
BATCH_SIZE = 8


TRAIN_DIR = "dataset/processed_multiclass/train"
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)

# Load Best Model
model = load_model("models/dr_multiclass_model.h5")

# Validation Dataset
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)

validation_data = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

print("\nEvaluating Model...\n")

validation_data.reset()

predictions = model.predict(validation_data)

predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_data.classes

class_labels = list(validation_data.class_indices.keys())

print(classification_report(
    true_classes,
    predicted_classes,
    target_names=class_labels
))

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "confusion_matrix.png"
    )
)

plt.show()

print("\n✅ Evaluation Completed Successfully!")