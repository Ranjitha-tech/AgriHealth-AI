
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array

# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_DIR / "models" / "plant_disease_model.keras"
CLASS_NAMES_PATH = PROJECT_DIR / "models" / "class_names.json"
IMAGE_PATH = PROJECT_DIR / "sample_images" / "test_leaf.JPG"

IMG_SIZE = (224, 224)

# --------------------------------------------------
# 2. Check required files
# --------------------------------------------------

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: {CLASS_NAMES_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(f"Test image not found: {IMAGE_PATH}")

# --------------------------------------------------
# 3. Load class names
# --------------------------------------------------

with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
    class_names = json.load(file)

# --------------------------------------------------
# 4. Load trained model
# --------------------------------------------------

print("\nLoading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# --------------------------------------------------
# 5. Prepare the leaf image
# --------------------------------------------------

print("\nLoading leaf image...")

image = load_img(IMAGE_PATH, target_size=IMG_SIZE)
image_array = img_to_array(image)

# Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
image_array = np.expand_dims(image_array, axis=0)

# Use the same preprocessing as MobileNetV2 training
image_array = preprocess_input(image_array)

# --------------------------------------------------
# 6. Predict plant disease
# --------------------------------------------------

print("Predicting disease...")

predictions = model.predict(image_array, verbose=0)[0]

if len(predictions) != len(class_names):
    raise ValueError(
        "The number of model outputs does not match "
        "the number of saved class names."
    )

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index]) * 100

# --------------------------------------------------
# 7. Display results
# --------------------------------------------------

print("\n" + "=" * 45)
print("       PLANT DISEASE PREDICTION")
print("=" * 45)

print(f"Image: {IMAGE_PATH.name}")
print(f"Predicted class: {predicted_class}")
print(f"Model confidence: {confidence:.2f}%")

print("\nTop 3 predictions:")

top_indices = np.argsort(predictions)[-3:][::-1]

for rank, index in enumerate(top_indices, start=1):
    print(
        f"{rank}. {class_names[index]}: "
        f"{predictions[index] * 100:.2f}%"
    )

print("=" * 45)
print("Prediction completed!")