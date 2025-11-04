# script/train_model.py
import os
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def autoencoder_data_generator(generator):
    """Wraps an ImageDataGenerator to yield (x, x) for autoencoder training."""
    for batch in generator:
        yield (batch, batch)

if __name__ == "__main__":
    params = load_params()
    img_height = params["training"]["img_height"]
    img_width = params["training"]["img_width"]
    batch_size = params["training"]["batch_size"]
    epochs = params["training"]["epochs"]

    processed_dir = "data/processed"
    model_out = "models/autoencoder_model.h5"
    os.makedirs("models", exist_ok=True)

    print(" Loading processed images for autoencoder training...")

    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        processed_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        processed_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        subset="validation"
    )

    print("Building autoencoder model...")

    input_img = Input(shape=(img_height, img_width, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = MaxPooling2D((2, 2), padding="same")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    encoded = MaxPooling2D((2, 2), padding="same")(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation="relu", padding="same")(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation="sigmoid", padding="same")(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss="mse")

    autoencoder.summary()

    checkpoint = ModelCheckpoint(
        model_out,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        mode="min"
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    print(" Starting autoencoder training...")
    autoencoder.fit(
        autoencoder_data_generator(train_gen),
        validation_data=autoencoder_data_generator(val_gen),
        epochs=epochs,
        steps_per_epoch=len(train_gen),
        validation_steps=len(val_gen),
        callbacks=[checkpoint, early_stop]
    )

    print(f" Autoencoder training complete. Model saved at {model_out}")
