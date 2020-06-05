import os
import sys

# Machine Learning Libs
from PIL import Image
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Global vars
EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
TEST_SIZE = 0.4

def load(dir):
    """
    Loads all images from a specified directory
    """
    # Initalize return arrays
    images = []
    labels = []

    # Iterate Dog and Cat directory
    for animal in os.listdir(dir):

        # Retrieve path to folder
        pathFolder = os.path.join(dir, animal)

        # Iterate every image in the directory
        for image in os.listdir(pathFolder):

            
            # Retrieve path to image
            img_path = os.path.join(pathFolder, image)

            # Pull the image and convert to grayscale
            img0 = Image.open(img_path)
            img1 = img0.convert(mode='L')

            # Resize to IMG_HEIGHT and IMG_WIDTH
            dim = (IMG_HEIGHT, IMG_WIDTH)
            img = img1.resize(dim)

            # Append image to images if correctly resized
            img = np.asarray(img)
            img = np.reshape(img, (IMG_WIDTH, IMG_HEIGHT, 1))
            if img.shape == (IMG_WIDTH, IMG_HEIGHT, 1):
                images.append(img)

                # Append 1 for cat, 0 for dog
                if animal == "Cat":
                    labels.append(1)
                else:
                    labels.append(0)

    
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """

    # Add layers to Sequential Model
    model = tf.keras.models.Sequential([

        # Convolutional layer
        tf.keras.layers.Conv2D(
            50, (3,3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Convolutional layer
        tf.keras.layers.Conv2D(
            50, kernel_size=(3, 3), activation="relu", strides=(1,1), padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),

        # Max-Pooling Layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(0.25),

        # Convolutional layer
        tf.keras.layers.Conv2D(
            100, kernel_size=(3, 3), activation="relu", strides=(1,1), padding="same", input_shape=(IMG_WIDTH, IMG_HEIGHT, 1)
        ),


        # 2nd Max-Pooling Layer, using 2x2 pool size
        tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
        tf.keras.layers.Dropout(.25),

        # Flatten units for input
        tf.keras.layers.Flatten(),


        # Add hidden layers
        tf.keras.layers.Dense(500, activation="relu"),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(250, activation="relu"),
        tf.keras.layers.Dropout(0.2),

        # Add an output layer
        tf.keras.layers.Dense(2, activation="softmax")

    ])

    # Compile the model with weight optimization, loss algorithms and emphasize accuracy
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model




def main():

    # Get image arrays and labels for all image files
    images, labels = load("PetImages")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.asarray(images), np.asarray(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        model.save(filename)
        print(f"Model saved to {filename}.")

if __name__ == "__main__":
    main()