import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import itertools

# ---------------- CONFIG ----------------
IMG_SIZE = 224
BATCH = 16
DATA_DIR = "dataset"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "currency_mobilenet_model.h5")
REPORT_DIR = "reports"
EPOCHS = 12

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ---------------- CONFUSION MATRIX PLOTTER ----------------
def plot_confusion_matrix(cm, classes, out_path=None):
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, str(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path)
    plt.close()

# ---------------- DATA GENERATORS ----------------
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.20,
    shear_range=0.15,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode="nearest"
)

val_test_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    os.path.join(DATA_DIR, "training"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)

val_gen = val_test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "validation"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    shuffle=True
)

test_gen = val_test_gen.flow_from_directory(
    os.path.join(DATA_DIR, "testing"),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH,
    class_mode="binary",
    shuffle=False
)

print(f"train: {train_gen.samples}, val: {val_gen.samples}, test: {test_gen.samples}")

# ---------------- BUILD MODEL (MobileNetV2) ----------------
def build_mobilenet_model():
    base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet")
    base.trainable = False  # freeze base

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

model = build_mobilenet_model()
model.summary()

# ---------------- TRAIN ----------------
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1
)

# ---------------- FINE TUNE ----------------
model.layers[0].trainable = True

for layer in model.layers[0].layers[:80]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5,
    verbose=1
)

# ---------------- EVALUATE ----------------
print("\nEvaluating on test set...")
loss, acc = model.evaluate(test_gen)
print(f"Test Accuracy: {acc * 100:.2f}%")

# ---------------- PREDICT ----------------
y_true = test_gen.classes
y_pred_prob = model.predict(test_gen)
y_pred = (y_pred_prob > 0.5).astype("int32").flatten()

cm = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:\n", cm)

cm_path = os.path.join(REPORT_DIR, "confusion_matrix.png")
plot_confusion_matrix(cm, ["real", "fake"], out_path=cm_path)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["real", "fake"]))

# ---------------- SAVE ----------------
model.save(MODEL_PATH)
print(f"\nSaved model at: {MODEL_PATH}")
