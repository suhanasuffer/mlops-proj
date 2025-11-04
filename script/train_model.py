# script/train_model.py
import os, yaml, numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)

def build_autoencoder(input_shape):
    input_img = layers.Input(shape=input_shape)
    # Encoder
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2,2), padding='same')(x)
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2,2), padding='same')(x)
    # Decoder
    x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2,2))(x)
    decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

if __name__ == "__main__":
    params = load_params()
    h = params["training"]["img_height"]
    w = params["training"]["img_width"]
    batch = params["training"]["batch_size"]
    epochs = params["training"]["epochs"]
    lr = params["training"]["learning_rate"]
    model_out = params["training"]["model_output"]

    # Data generator: assumes processed_dir contains images (optionally in subfolders)
    datagen = ImageDataGenerator(rescale=1./255)
    processed_dir = params["data"].get("processed_dir", "data/processed")
    train_gen = datagen.flow_from_directory with class_mode=None for autoencoder
    train_gen = datagen.flow_from_directory(
        processed_dir,
        target_size=(h, w),
        batch_size=batch,
        class_mode=None,    # no labels
        shuffle=True
    )
    # steps per epoch
    steps = max(1, train_gen.samples // batch)

    input_shape = (h, w, 3)
    autoencoder = build_autoencoder(input_shape)

    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    # Callbacks: early stopping + model checkpoint
    es = EarlyStopping(monitor='loss', patience=params["training"].get("early_stopping_patience", 5), restore_best_weights=True)
    ckpt = ModelCheckpoint(model_out, save_best_only=True, monitor='loss')

    autoencoder.fit(train_gen, epochs=epochs, steps_per_epoch=steps, callbacks=[es, ckpt])
    # final save if not already saved
    if not  os.path.exists(model_out):
        autoencoder.save(model_out)
    print("Model saved to:", model_out)
