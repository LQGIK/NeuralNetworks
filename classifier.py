import cv2
import os
import sys

import tensorflow as tf
import numpy as np




def main():

    # Initialize the classifier
    cascPath = os.path.dirname(cv2.__file__)+"/data/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Load model and error checking
    if len(sys.argv) != 2:
        print("Usage: python classifier.py <model_name.h5>")
        return

    model_name = sys.argv[1]
    model = tf.keras.models.load_model(model_name)
    if not model:
        print("This isn't an acceptable model!")
        return


    # Capture video footage
    video_capture = cv2.VideoCapture(0)


    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()

        # Convert to greyscale cv2 image
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        # Identify faces in image (gray), 
        faces = faceCascade.detectMultiScale (
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        

        # Reshape and predict
        dim = (30, 30)
        small_frame = cv2.resize(gray, (0, 0), fx=0.046875, fy=0.0625)
        image = small_frame.reshape(dim)
        image = [np.array(image).reshape(1, 30, 30, 1)]
        prediction = model.predict(image).argmax()





        # Draw a rectangle around the faces
        # Sets x,y,w,h to dimensions of face, and draws green rectangle
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow("Video", frame)




        
        if cv2.waitKey(1) & 0xFF == ord("i"):
            if prediction == 1:
                print("\nHello Jamie!\n")
            else:
                print("\nHello Joon!\n")
        if cv2.waitKey(2) & 0xFF == ord("q"):
            break


    # Release capture frames
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()