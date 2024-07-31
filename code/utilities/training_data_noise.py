"""
Script to apply various image transformations to a dataset of alphanumeric characters.
Transforms include Gaussian noise, Gaussian blur, rotation, translation, and upscaling.
The transformed images are saved to a new directory for use in training machine learning models.

Directory structure:
- This script should be placed in jetson-ocr/code/utilities.
- The original dataset should be in jetson-ocr/data/dataset.
- The modified dataset will be saved to jetson-ocr/data/dataset_modified.

Dependencies:
- numpy
- opencv-python

Usage:
1. Ensure the original dataset is in the correct directory.
2. Run this script from the command line:
   python3 training_data_noise.py

The script will process each image in the dataset, apply transformations, and save the modified images.
"""

import numpy as np
import cv2
import random
import os

# Applies Gaussian Noise to the image
def add_gaussian_noise(image, mean=0, std=25):
    # Generate Gaussian noise with specified mean and standard deviation
    gauss = np.random.normal(mean, std, image.shape).astype('float32')
    noisy_image = image.astype('float32') + gauss

    # Clip pixel values to ensure they are within the valid range [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')

    return noisy_image


# Applies Gaussian Blur to the image
def add_gaussian_blur(image, kernel_size=(5, 5)):
    return cv2.GaussianBlur(image, kernel_size, 0)


# Rotates the image by a random angle between -10 and 10 degrees
def rotate_image(image):
    height, width = image.shape[:2]

    # Generate random angle between -10 and 10 degrees
    angle = random.randint(-10, 10)

    # Calculate the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

    # Apply rotation using warpAffine
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), borderValue=(255, 255, 255))

    return rotated_image


# Centers the characters to middle of frame
def translate_image(image):
    # Define translation matrix
    # Shift 6 pixels right and 5 pixels down
    dx = 6
    dy = 5
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Get the dimensions of the image
    rows, cols = image.shape[:2]

    # Translate image and set background to white
    translated_image = cv2.warpAffine(image, translation_matrix, (cols, rows), borderValue=[255, 255, 255])

    # Upscales the translated images to 1.25 original size
    translated_image = upscale_image(translated_image)

    return translated_image


# Upscales image to 1.25 of original scale
def upscale_image(image):
    # Get the dimensions of the image
    height, width = image.shape[:2]

    zoom_factor = 1.25

    # Calculate the center of the image
    center_x, center_y = width // 2, height // 2

    # Calculate the cropping coordinates
    crop_w = int(width / zoom_factor)
    crop_h = int(height / zoom_factor)
    x1 = max(0, center_x - crop_w // 2)
    x2 = min(width, center_x + crop_w // 2)
    y1 = max(0, center_y - crop_h // 2)
    y2 = min(height, center_y + crop_h // 2)

    # Crop the center of the image
    cropped = image[y1:y2, x1:x2]

    # Resize the cropped image back to original dimensions
    zoomed_image = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)

    return zoomed_image


def main():
    # Get the current working directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the base directory as two levels up from the script directory (jetson-ocr)
    base_dir = os.path.abspath(os.path.join(script_dir, "../../"))

    # Define the current and new directories relative to the base directory
    current_directory_path = os.path.join(base_dir, "data", "dataset")
    new_directory_path = os.path.join(base_dir, "data", "dataset_modified")

    # Process all the folders in a directory
    for foldername in os.listdir(current_directory_path):
        # Get current folder path and construct new folder path
        current_folder_path = os.path.join(current_directory_path, foldername)
        new_folder_path = os.path.join(new_directory_path, foldername)

        # Process all the files in a folder
        for filename in os.listdir(current_folder_path):
            # Get current file path and construct new file path
            current_file_path = os.path.join(current_folder_path, filename)
            new_file_path = os.path.join(new_folder_path, filename)

            image = cv2.imread(current_file_path)
            
            # Generate random values to apply noise and/or rotation
            noise = random.randint(0, 100)
            rotation = random.randint(0, 1)

            # Translate and upsacle 100% of the images
            image = translate_image(image)

            # Apply a rotation to 50% of the images
            if rotation == 1:
                image = rotate_image(image)

            # Apply noise to 30% of the images
            if noise > 0 and noise < 15: # Gaussian Noise - 15%
                image = add_gaussian_noise(image)
            elif noise > 15 and noise < 30: # Gaussian Blur - 15%
                image = add_gaussian_blur(image)

            # Create the new folder
            try:
                os.makedirs(new_folder_path, exist_ok=True)
            except Exception as e:
                print(f"Error creating folder '{new_folder_path}': {e}")

            # Write the processed image to the new location 
            cv2.imwrite(new_file_path, image)

if __name__ == '__main__':
    main()