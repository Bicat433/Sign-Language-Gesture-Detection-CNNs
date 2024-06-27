import cv2 as cv
import mediapipe as mp
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def create_directories(directory_names):
    for directory_name in directory_names:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

hand_signs = ["A", "B", "C", "D", "E", "F", "G"]
base_dir = "D:/Python Projects/Sign Language Gesture Detection/"
dataset_dir = os.path.join(base_dir, "dataset")

create_directories([dataset_dir])
hand_sign_dirs = [os.path.join(dataset_dir, sign) for sign in hand_signs]
create_directories(hand_sign_dirs)

# Change the video capture source to the IP webcam URL
capture = cv.VideoCapture("http://10.152.221.89:8080/video")

for sign in hand_signs:
    image_count = 0
    print(f"Capturing images for sign: {sign}. Press 's' to save, 'q' to quit.")
    
    while image_count < 800:
        ret, frame = capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert the image from BGR to RGB
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = hands.process(frame_rgb)
        
        # Convert the image back to BGR for display
        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Resize the frame to make the window shorter and narrower (half the width and height)
        frame_bgr_resized = cv.resize(frame_bgr, (frame_bgr.shape[1] // 2, frame_bgr.shape[0] // 2))
        
        cv.imshow("Frame", frame_bgr_resized)
        key = cv.waitKey(1) & 0xFF
        
        if key == ord('s') and results.multi_hand_landmarks:
            image_path = os.path.join(dataset_dir, sign, f"hand_sign_{image_count+1}.jpg")
            cv.imwrite(image_path, frame_bgr)
            image_count += 1
            print(f"Image {image_count} for {sign} saved.")
        
        elif key == ord('q'):
            break

capture.release()
cv.destroyAllWindows()
hands.close()




# capture = cv.VideoCapture(0)
