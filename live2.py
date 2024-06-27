import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import mediapipe as mp
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Define the directory path
dir_path = os.path.realpath('D:\\Python Projects\\Sign Language Gesture Detection')

# Load the trained model
model_path = os.path.join(dir_path, 'sign_language_model2.h5')

# Debugging: Print out the resolved model path>> had issues without using os, and directly loading the model 
st.write(f"Resolved model path: {model_path}")

# Ensure the model file exists at the specified path
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No file or directory found at {model_path}")

model = tf.keras.models.load_model(model_path)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define the preprocess function for the images
def preprocess_for_model(img, size=(64, 64)):
    img = cv.resize(img, size)  # Resize image to match model input
    img = img / 255.0  # Normalize pixel values to [0, 1]
    return img.reshape(1, *size, 3)  # Reshape for the model

# Define the model classes
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

# Video transformer class for sign language detection
class SignLanguageTransformer(VideoTransformerBase):
    def __init__(self):
        self.hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        frame_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Assuming the hand landmarks can be used to get a bounding box (Media pipe sol)
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min, x_max = min(x_coords) * img.shape[1], max(x_coords) * img.shape[1]
                y_min, y_max = min(y_coords) * img.shape[0], max(y_coords) * img.shape[0]
                x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)

                # Crop and preprocess the image for the model
                hand_img = img[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue
                hand_img_processed = preprocess_for_model(hand_img)

                # Predict the hand sign
                prediction = model.predict(hand_img_processed)
                predicted_class = classes[np.argmax(prediction)]
                cv.putText(img, predicted_class, (x_min, y_min - 10), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        return img

# URL of the background image
background_image_url = 'https://static1.cbrimages.com/wordpress/wp-content/uploads/2019/02/naruto-hand-signs.jpg'

# Login screen
def login():
    st.markdown(f"""
        <style>
            .login-container {{
                background: url('{background_image_url}') no-repeat center center fixed; 
                background-size: cover;
                padding: 50px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                text-align: center;
            }}
            .login-title {{
                font-size: 2.5em;
                color: #4a90e2;
                margin-bottom: 20px;
            }}
            .login-input {{
                font-size: 1.2em;
                padding: 10px;
                margin: 10px 0;
                border-radius: 5px;
                border: 1px solid #ccc;
                width: 100%;
            }}
            .login-button {{
                background-color: #4a90e2;
                color: white;
                font-size: 1.2em;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
            }}
            .login-button:hover {{
                background-color: #357ABD;
            }}
        </style>
        <div class="login-container">
            <div class="login-title">Login</div>
    """, unsafe_allow_html=True)
    
    username = st.text_input("", placeholder="Username", key="username", help="Enter your username")
    password = st.text_input("", placeholder="Password", type="password", key="password", help="Enter your password")
    
    if st.button("Login", key="login_button"):
        if username == "admin" and password == "password":  # pw and user name 
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid credentials")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main application
def main():
    st.markdown(f"""
        <style>
            .main-container {{
                background: url('{background_image_url}') no-repeat center center fixed; 
                background-size: cover;
                padding: 50px;
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                text-align: center;
            }}
            .main-title {{
                font-size: 2.5em;
                color: #4a90e2;
                margin-bottom: 20px;
            }}
            .main-description {{
                font-size: 1.2em;
                color: #333;
                margin-bottom: 40px;
            }}
        </style>
        <div class="main-container">
            <div class="main-title">Real-time Sign Language Detection</div>
            <div class="main-description">
                This application uses a trained model to detect and classify hand signs in real-time.
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    webrtc_streamer(key="example", video_transformer_factory=SignLanguageTransformer)

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main()
else:
    login()
