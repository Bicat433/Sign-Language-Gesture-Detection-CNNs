import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('sign_language_model5.h5')

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define the preprocess function for the images
def preprocess_for_model(img, size=(64, 64)):
    img = cv.resize(img, size)  # Resize image to match model input
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img.reshape(1, *size, 3)  # Reshape for the model

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']  # Your model classes

# Initialize webcam
cap = cv.VideoCapture("http://192.168.10.50:8080/video")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Assuming the hand landmarks can be used to get a bounding box (mp sol)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_coords) * frame.shape[1], max(x_coords) * frame.shape[1]
            y_min, y_max = min(y_coords) * frame.shape[0], max(y_coords) * frame.shape[0]
            x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

            # Crop and preprocess the image for the model
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue
            hand_img_processed = preprocess_for_model(hand_img)

            # Predict the hand sign
            prediction = model.predict(hand_img_processed)
            predicted_class = classes[np.argmax(prediction)]
            cv.putText(frame, predicted_class, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

    # Display the frame
    cv.imshow('Sign Language Detection', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
hands.close()
