import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "static")
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2


def load_data():
    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    generator = ImageDataGenerator(rescale=1.0 / 255, validation_split=VALIDATION_SPLIT)
    train_generator = generator.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        seed=42,
    )
    validation_generator = generator.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42,
    )
    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("The train folder must contain images in class subfolders.")

    class_names = [name for name, _ in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
    with open(os.path.join(BASE_DIR, "class_names.pkl"), "wb") as class_file:
        pickle.dump(class_names, class_file)
    return train_generator, validation_generator, class_names


def build_model(class_count):
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False
    features = GlobalAveragePooling2D()(base_model.output)
    features = Dense(512, activation="relu")(features)
    predictions = Dense(class_count, activation="softmax")(features)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_graphs(history):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for metric, ylabel, filename in (
        ("accuracy", "Accuracy", "accuracy_plot.png"),
        ("loss", "Loss", "loss_plot.png"),
    ):
        plt.figure()
        plt.plot(history.history[metric], label=f"Training {ylabel}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {ylabel}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()


def evaluate_model(model, validation_generator, class_names):
    validation_generator.reset()
    probabilities = model.predict(validation_generator, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print("Validation confusion matrix:")
    print(confusion_matrix(validation_generator.classes, predictions))
    print("Validation classification report:")
    print(
        classification_report(
            validation_generator.classes,
            predictions,
            labels=list(range(len(class_names))),
            target_names=class_names,
            zero_division=0,
        )
    )

    # Use a separate test split only when it has the same class folders and images.
    if not os.path.isdir(TEST_DIR):
        return
    test_generator = ImageDataGenerator(rescale=1.0 / 255).flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    if test_generator.samples == 0:
        print("No images in test/: skipped held-out test evaluation.")
        return
    if test_generator.class_indices != validation_generator.class_indices:
        raise ValueError("test/ must use the same class subfolders as train/.")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Held-out test accuracy: {test_accuracy:.4f} (loss {test_loss:.4f})")


if __name__ == "__main__":
    training, validation, names = load_data()
    model = build_model(len(names))
    history = model.fit(training, epochs=10, validation_data=validation)
    plot_graphs(history)
    model.save(os.path.join(BASE_DIR, "ResNet50_model.h5"))
    evaluate_model(model, validation, names)
