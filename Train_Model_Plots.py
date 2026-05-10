import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")

DATASET_DIR = r"C:\Users\shaik\Downloads\Retinal\dataset"

IMG_SIZE = (160, 160)
BATCH = 64
EPOCHS = 15

MODEL_OUT = "models/cnn_retinal.keras"
OUTPUT_DIR = "Model_Plots_Samples"

os.makedirs("models", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_ds = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_ds = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_indices = train_ds.class_indices
with open("models/class_indices.json", "w") as f:
    json.dump(class_indices, f)

class_names = list(class_indices.keys())

base = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_shape=(160, 160, 3)
)

base.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(3e-4),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1
)

model.save(MODEL_OUT)

plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy")
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy.png"))
plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss")
plt.savefig(os.path.join(OUTPUT_DIR, "loss.png"))
plt.show()

val_ds.reset()
preds = model.predict(val_ds, verbose=0)
y_pred = np.argmax(preds, axis=1)
y_true = val_ds.classes

report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

with open(os.path.join(OUTPUT_DIR, "classification_report.json"), "w") as f:
    json.dump(report, f, indent=4)

print(classification_report(y_true, y_pred, target_names=class_names))

def predict(img_path):
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_arr = np.array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)

    pred = model.predict(img_arr, verbose=0)[0]
    idx = np.argmax(pred)

    return class_names[idx], pred[idx]

sample_images = []

for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".png", ".jpeg")):
            sample_images.append(os.path.join(root, f))
        if len(sample_images) >= 10:
            break

for i, img_path in enumerate(sample_images):
    label, conf = predict(img_path)

    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)

    plt.figure()
    plt.imshow(img)
    plt.title(f"{label} ({conf*100:.2f}%)")
    plt.axis("off")

    save_path = os.path.join(OUTPUT_DIR, f"sample_{i+1}.png")
    plt.savefig(save_path)
    plt.show()

print("Done")