import pickle
import numpy as np
import os
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split



def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load(dir):
    """
    Loads all files from a directory
    """

    images = []
    labels = []
    label_names = []

    for file in os.listdir(dir):

        # Get path to file
        img_path = os.path.join(dir, file)

        # Pull the image
        batch = unpickle(img_path)

        # Sort into labels or images
        if file == "batches.meta":
            label_names = batch
        else:

            # Append images
            for row in batch[b'data']:

                # Organize pixels
                red = row[:1024]
                green = row[1024:2048]
                blue = row[2048:]

                # Zip and append
                pixels = np.array(list(zip(red, green, blue)))
                image = np.reshape(pixels, (32, 32, 3))
                images.append(image)

            # Append labels
            for label in batch[b'labels']:
                labels.append(label)


    return np.array(images), np.array(labels), label_names

def get_model():
    """
    Returns Neural Network Model
    """
    model = tf.keras.models.Sequential([

        # Convolutional Layer with 50 filters
        tf.keras.layers.Conv2D(
            50, kernel_size=(3, 3), strides=(1,1), padding="same", activation="relu", input_shape=(32, 32, 3)
        ),

        # 2nd Convolutional Layer with 75 filters
        tf.keras.layers.Conv2D(
            75, kernel_size=(3, 3), strides=(1,1), padding="same", activation="relu", input_shape=(32, 32, 3)
        ),

        # Max Pooling with dropout of .25
        tf.keras.layers.MaxPool2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),

        # 3rd Conv Layer with 125 filters
        tf.keras.layers.Conv2D(
            125, kernel_size=(3,3), strides=(1,1), padding="same", activation="relu"
        ),
        
        # MaxPooling Layer with dropout of .25 after
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),
        tf.keras.layers.Dropout(0.25),

        # Flatten units for input into network
        tf.keras.layers.Flatten(),

        # Dense Hidden Feed Forward Layers
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dropout(0.3),

        # Output layer
        tf.keras.layers.Dense(10, activation="softmax")

    ])

    # Compile model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    
    # Load file into memory
    images, labels, label_names = load("cifar-10-batches-py")
    
    # Split and prepare data
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.5
    )

    # Create model
    model = get_model()

    # Train model
    model.fit(x_train, y_train, epochs=10)

    # Evaluate Neural Net
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


if __name__ == "__main__":
    main()