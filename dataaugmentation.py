import cv2 as cv
import os
import numpy as np

def create_directories(directory_names):
    for directory_name in directory_names:
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

def rotate_image(image, angle):
    """Rotate the image by the given angle."""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def preprocess_image(image):
    """Resize and convert the image to float32."""
    # Resize the image to 256x256 using INTER_CUBIC for better quality
    resized_image = cv.resize(image, (256, 256), interpolation=cv.INTER_CUBIC)
    float_image = resized_image.astype(np.float32)  # Convert image to float32 for compatibility with normalization
    return float_image

def augment_images(input_dir, output_dir):
    """Apply augmentations to images in the input directory and save them to the output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):  # Check for jpg files
            img_path = os.path.join(input_dir, filename)
            img = cv.imread(img_path)
            if img is None:
                continue

            # Preprocess the image before augmentation
            preprocessed_img = preprocess_image(img)

            # Normalize the image for saving and processing
            normalized_img = preprocessed_img / 255.0

            # Grayscale conversion
            gray_img = cv.cvtColor(preprocessed_img, cv.COLOR_BGR2GRAY)
            gray_img_path = os.path.join(output_dir, f"gray_{filename}")
            cv.imwrite(gray_img_path, gray_img)  # Save gray image directly

            # Rotations at smaller angles
            for angle in [-30, 30]:
                rotated_img = rotate_image(normalized_img, angle)
                rotated_img_path = os.path.join(output_dir, f"rotated_{angle}_{filename}")
                cv.imwrite(rotated_img_path, rotated_img * 255)  # Multiply by 255 to convert back to uint8 for saving
                """
                Transformation and greyscales 
                """

# Directory setup
base_dir = "D:/Python Projects/Sign Language Gesture Detection/"
dataset_dir = os.path.join(base_dir, "dataset")
augmented_dir = os.path.join(base_dir, "augmented_dataset")

# Ensure the augmented directory structure is prepared
create_directories([augmented_dir])

hand_signs = ["A", "B", "C", "D", "E", "F", "G"]
for sign in hand_signs:
    input_sign_dir = os.path.join(dataset_dir, sign)
    output_sign_dir = os.path.join(augmented_dir, sign)
    create_directories([output_sign_dir])
    augment_images(input_sign_dir, output_sign_dir)
    print(f"Augmentation complete for sign: {sign}")
