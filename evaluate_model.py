import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ==============================
# 1. Project paths
# ==============================

MODEL_PATH = "models/plant_disease_model.keras"
CLASS_NAMES_PATH = "models/class_names.json"
DATASET_PATH = "dataset/PlantVillage"

RESULTS_PATH = "results"

os.makedirs(RESULTS_PATH, exist_ok=True)


# ==============================
# 2. Load trained model
# ==============================

print("Loading trained model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")


# ==============================
# 3. Load class names
# ==============================

with open(CLASS_NAMES_PATH, "r") as file:
    class_names = json.load(file)

print(f"Number of classes: {len(class_names)}")


# ==============================
# 4. Load validation dataset
# ==============================

print("Loading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=(224, 224),
    batch_size=32
)


# ==============================
# 5. Evaluate model
# ==============================

print("\nEvaluating model...")

loss, accuracy = model.evaluate(
    validation_dataset,
    verbose=1
)

print("\n==============================")
print("MODEL EVALUATION RESULTS")
print("==============================")

print(f"Validation Loss     : {loss:.4f}")
print(f"Validation Accuracy : {accuracy * 100:.2f}%")

print("==============================")


# ==============================
# 6. Save evaluation results
# ==============================

with open(
    os.path.join(RESULTS_PATH, "evaluation_results.txt"),
    "w"
) as file:

    file.write("MODEL EVALUATION RESULTS\n")
    file.write("========================\n")
    file.write(f"Validation Loss     : {loss:.4f}\n")
    file.write(f"Validation Accuracy : {accuracy * 100:.2f}%\n")


# ==============================
# 7. Generate predictions
# ==============================

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in validation_dataset:

    predictions = model.predict(
        images,
        verbose=0
    )

    predicted_classes = np.argmax(
        predictions,
        axis=1
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_classes)

y_true = np.array(y_true)
y_pred = np.array(y_pred)


# ==============================
# 8. Classification report
# ==============================

print("\nGenerating classification report...")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    zero_division=0
)

print("\n")
print(report)


with open(
    os.path.join(RESULTS_PATH, "classification_report.txt"),
    "w"
) as file:

    file.write(report)


# ==============================
# 9. Confusion matrix
# ==============================

print("Generating confusion matrix...")

cm = confusion_matrix(
    y_true,
    y_pred
)

fig, ax = plt.subplots(
    figsize=(14, 12)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(
    ax=ax,
    xticks_rotation=90,
    cmap="Blues",
    colorbar=False
)

plt.title(
    "Plant Disease Classification - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_PATH,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

print("Confusion matrix saved!")


# ==============================
# 10. Finished
# ==============================

print("\n================================")
print("STEP 20 COMPLETED SUCCESSFULLY!")
print("================================")

print("\nResults saved in:")
print("results/")