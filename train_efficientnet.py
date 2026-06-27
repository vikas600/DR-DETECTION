import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization
)

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 20

TRAIN_DIR = "dataset/processed_multiclass/train"

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,

    validation_split=0.2,

    rotation_range=20,

    zoom_range=0.2,

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=True,

    fill_mode="nearest"
)


train_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

validation_data = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)
# -----------------------------
# Compute Class Weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.classes),
    y=train_data.classes
)

class_weights = dict(enumerate(class_weights))

print("\nClass Weights:")
print(class_weights)

print("\nClass Mapping")
print(train_data.class_indices)


# -----------------------------
# Load EfficientNetB0
# -----------------------------
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

# Freeze pretrained layers
base_model.trainable = False

print("EfficientNetB0 loaded successfully!")


# -----------------------------
# Custom Classification Head
# -----------------------------
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = BatchNormalization()(x)

x = Dropout(0.4)(x)

x = Dense(256, activation="relu")(x)

x = Dropout(0.3)(x)

predictions = Dense(5, activation="softmax")(x)

model = Model(
    inputs=base_model.input,
    outputs=predictions
)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "models/dr_multiclass_model.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1
)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1
)

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        early_stop,
        checkpoint,
        reduce_lr
    ]
)



# ==========================================
# STAGE 2 : Fine Tuning
# ==========================================

print("\nStarting Fine Tuning...\n")

# Unfreeze the base model
base_model.trainable = True

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# 👇 YAHAN ADD KARO
fine_tune_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

history_finetune = model.fit(
    train_data,
    validation_data=validation_data,
    initial_epoch=20,
    epochs=25,
    class_weight=class_weights,
    callbacks=[
        fine_tune_stop,   # 👈 yahan early_stop ki jagah ye
        checkpoint,
        reduce_lr
    ]
)

model.save("models/dr_multiclass_model_finetuned.h5")

print("✅ Fine Tuned Model Saved Successfully!")


