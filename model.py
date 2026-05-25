import tensorflow as tf
from tensorflow.keras import layers, models

def create_model():
    model = models.Sequential()

    # Input + Conv Layer 1
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Conv Layer 2
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Conv Layer 3
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(2,2))

    # Flatten
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))   # NEW ADDITION

    # Output Layer
    model.add(layers.Dense(3, activation='softmax'))

    return model


model = create_model()
model.summary()