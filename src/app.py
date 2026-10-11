from __future__ import annotations

import random
import shutil
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
DATASET_URL = "https://storage.googleapis.com/datascience-materials/dogs-vs-cats.zip"


def download_dataset() -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "dogs-vs-cats.zip"

    if not zip_path.exists():
        print(f"Descargando dataset desde {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, zip_path)

    extracted_dir = RAW_DIR / "dogs-vs-cats"
    if not extracted_dir.exists():
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(RAW_DIR)

    candidates = [
        RAW_DIR / "train",
        extracted_dir / "train",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    nested_train = list(RAW_DIR.glob("**/train"))
    if nested_train:
        return nested_train[0]

    raise FileNotFoundError("No se encontró la carpeta train del dataset.")


def split_dataset(source_dir: Path, max_per_class: int = 3000):
    train_dir = PROCESSED_DIR / "train"
    val_dir = PROCESSED_DIR / "validation"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for label in ("cat", "dog"):
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (val_dir / label).mkdir(parents=True, exist_ok=True)

        files = sorted(source_dir.glob(f"{label}*.jpg"))
        random.shuffle(files)
        files = files[:max_per_class]

        split_index = max(1, int(len(files) * 0.8))
        for i, file_path in enumerate(files):
            destination = train_dir / label / file_path.name if i < split_index else val_dir / label / file_path.name
            shutil.copy2(file_path, destination)

    return train_dir, val_dir


def show_sample_images(data_dir: Path, label: str):
    images = sorted((data_dir / label).glob("*.jpg"))[:9]
    if not images:
        raise FileNotFoundError(f"No hay imágenes para la clase {label} en {data_dir}")

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    for ax, image_path in zip(axes.flat, images):
        ax.imshow(plt.imread(image_path))
        ax.set_title(label.capitalize())
        ax.axis("off")

    fig.tight_layout()
    output_path = PROCESSED_DIR / f"{label}_samples.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Figura guardada en {output_path}")


def build_generators(train_dir: Path, val_dir: Path):
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen


def build_model():
    base_model = EfficientNetB0(include_top=False, weights=None, input_shape=(224, 224, 3))
    model = tf.keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(2, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    source_train_dir = download_dataset()
    train_dir, val_dir = split_dataset(source_train_dir, max_per_class=2000)

    for label in ("cat", "dog"):
        show_sample_images(train_dir, label)

    train_gen, val_gen = build_generators(train_dir, val_dir)
    model = build_model()

    checkpoint_path = MODEL_DIR / "best_dogs_vs_cats.keras"
    checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=5,
        callbacks=[checkpoint, early_stop],
        verbose=1,
    )

    best_model = tf.keras.models.load_model(checkpoint_path)
    loss, accuracy = best_model.evaluate(val_gen, verbose=1)
    print(f"Pérdida final: {loss:.4f}")
    print(f"Precisión final: {accuracy:.4f}")

    final_model_path = MODEL_DIR / "dogs_vs_cats_classifier.keras"
    best_model.save(final_model_path)
    print(f"Modelo guardado en {final_model_path}")

    print("Entrenamiento finalizado correctamente.")
    print(f"Historial de entrenamiento: {history.history.keys()}")


if __name__ == "__main__":
    main()
