import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = "dataset/PlantVillage"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_SPLIT = 0.2
EPOCHS = 10

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------------
# 2. Load the dataset
# --------------------------------------------------

print("Loading dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=VALIDATION_SPLIT,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nNumber of classes:", num_classes)
print("Classes:")

for index, class_name in enumerate(class_names):
    print(index, "->", class_name)

# --------------------------------------------------
# 3. Save class names
# --------------------------------------------------

with open("models/class_names.json", "w") as file:
    json.dump(class_names, file, indent=4)

# --------------------------------------------------
# 4. Improve dataset performance
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# --------------------------------------------------
# 5. Data augmentation
# --------------------------------------------------

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --------------------------------------------------
# 6. Load MobileNetV2
# --------------------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze the pretrained layers
base_model.trainable = False

# --------------------------------------------------
# 7. Build the model
# --------------------------------------------------

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.2)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax"
)(x)

model = models.Model(inputs, outputs)

# --------------------------------------------------
# 8. Compile the model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --------------------------------------------------
# 9. Callbacks
# --------------------------------------------------

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True
)

# --------------------------------------------------
# 10. Train the model
# --------------------------------------------------

print("\nStarting training...\n")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# --------------------------------------------------
# 11. Final evaluation
# --------------------------------------------------

print("\nEvaluating model...")

loss, accuracy = model.evaluate(validation_dataset)

print(f"\nValidation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")

print("\nTraining completed successfully!")

print("Model saved to:", MODEL_PATH)
print("Class names saved to: models/class_names.json")