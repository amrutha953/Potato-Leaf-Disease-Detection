import tensorflow as tf
from tensorflow.keras import layers, models

# CNN Model for Potato Leaf Disease Detection
def create_model():
    model = models.Sequential()

    # Input + First Conv Layer
    model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)))
    model.add(layers.MaxPooling2D(2,2))

    # Second Conv Layer
    model.add(layers.Conv2D(64, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Third Conv Layer
    model.add(layers.Conv2D(128, (3,3), activation='relu'))
    model.add(layers.MaxPooling2D(2,2))

    # Flatten
    model.add(layers.Flatten())

    # Fully Connected Layer
    model.add(layers.Dense(128, activation='relu'))

    # Output Layer (3 classes)
    model.add(layers.Dense(3, activation='softmax'))

    return model


# Create model
model = create_model()

# Show model summary
model.summary()