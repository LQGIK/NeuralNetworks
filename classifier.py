# Import sys and Image for image processing
import sys
from PIL import Image

# ML libs
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

# Load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot

# Global variables
IMG_WIDTH = 30
IMG_HEIGHT = 30


def preprocess(img_path):
    """
    Will preprocess image in grayscale, then resize for input into model
    """

    # Pull the image and convert to grayscale
    img0 = Image.open(img_path)
    img1 = img0.convert(mode='L')

    # Resize to IMG_HEIGHT and IMG_WIDTH
    dim = (IMG_HEIGHT, IMG_WIDTH)
    image = img1.resize(dim)

    # Append image to images if correctly resized
    image = [np.array(image).reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)]
    return image





def main():

    # Usage
    if len(sys.argv) < 3:
        print("Usage: python classifier.py <path_to_model> <image_path>")
        return

    # Load model and error checking
    model_name = sys.argv[1]
    model = tf.keras.models.load_model(model_name)
    if not model:
        print("This isn't an acceptable model!")
        return

    print("\n\n\n\n\nProcessing...")

    # Load and preprocess image if it is acceptable
    image = preprocess(sys.argv[2])    

    # Split data into training and testing sets
    prediction = model.predict(image).argmax()

    # Make prediction
    if prediction == 1:
        print("\nYou uploaded an image of a CAT!\n")
    else:
        print("\nYou uploaded an image of a DOG!\n")


    # Load image as pixel array
    imageDisp = Image.open(sys.argv[2])
    pyplot.imshow(imageDisp)
    pyplot.show()
    



if __name__ == "__main__":
    main()