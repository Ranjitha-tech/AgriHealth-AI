import os

DATASET_PATH = "dataset/PlantVillage"

classes = [
    folder
    for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(os.path.join(DATASET_PATH, folder))
]

print("Number of classes:", len(classes))

print("\nFirst 10 classes:")
for class_name in classes[:10]:
    print("-", class_name)

print("\nImages in the first class:")

if classes:
    first_class_path = os.path.join(DATASET_PATH, classes[0])

    images = [
        file
        for file in os.listdir(first_class_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print("Class:", classes[0])
    print("Number of images:", len(images))