import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import cv2
import numpy as np

# 1. Load the trained model
try:
    model = tf.keras.models.load_model('F:\\Spects_wear_or_not\\pythonProject\\glasses_detector.h5')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


# Check if camera is available
def check_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera is not available. Please check your camera connection.")
        return False
    cap.release()
    return True


# Preprocess the image
def preprocess_image(frame):
    img_resized = cv2.resize(frame, (224, 224))
    img_array = np.expand_dims(img_resized, axis=0) / 255.0
    return img_array


# Predict whether glasses are being worn
def predict_glasses(model, img_array):
    prediction = model.predict(img_array)
    return prediction[0][0] > 0.5


# Capture an image and detect glasses when spacebar is pressed
def detect_glasses_image():
    if not check_camera():
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the live camera feed
        cv2.imshow('Press SPACE to Capture', frame)

        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):  # Spacebar pressed
            img_array = preprocess_image(frame)

            # Prediction
            if predict_glasses(model, img_array):
                label = "Wearing Glasses"
            else:
                label = "Not Wearing Glasses"

            # Display the label on the frame
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the frame with the prediction
            cv2.imshow('Glasses Detection', frame)

            # Wait until a key is pressed to exit
            cv2.waitKey(0)
            print("Exiting...")
            break
    
        elif key == ord('q'):  # 'q' to quit
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()


# Example usage
detect_glasses_image()
