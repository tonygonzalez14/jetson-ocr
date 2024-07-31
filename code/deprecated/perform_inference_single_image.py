"""
This script performs single-character recognition using a CUDA-based inference model. The script:
1. Captures an image using a camera.
2. Extracts and processes the character from the captured image.
3. Performs CUDA-based inference to classify the character.
4. Overlays the classified character image onto the captured image.
5. Logs the process and errors.

Note: This program is no longer in use and was part of the development process.

Dependencies:
- OpenCV
- cvzone
- NumPy
- ctypes (for interfacing with the CUDA library)
- argparse (for command-line arguments)
- json (for configuration)

Usage:
1. Ensure the CUDA library (libcuda_inference.so) and configuration file are available.
2. Place the script in the directory with 'Test Images/captured_image.jpg' and 'letters_png/'.
3. Run the script with optional configuration file path.
"""

import cv2
import cvzone
import time
import numpy as np
import ctypes
import logging
import argparse
import json

# Set up basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a file handler
file_handler = logging.FileHandler('program.log', mode='w')
file_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)

def get_image_filename(character_index):
    """
    Returns the filename of the image corresponding to the character index.
    """
    if 0 <= character_index <= 9:
        return f"{character_index}.png"
    elif 10 <= character_index <= 35:  # Uppercase letters
        letter = chr(ord('A') + (character_index - 10))
        return f"{letter}_U.png"
    elif 36 <= character_index <= 61:  # Lowercase letters
        letter = chr(ord('a') + (character_index - 36))
        return f"{letter}_L.png"
    else:
        logger.error("Invalid character number")
        return "Invalid character number"

def capture_image(config):
    """
    Captures an image from the camera and saves it as 'captured_image.jpg'.
    """
    logger.info("Opening the camera")
    
    # Open the camera with GStreamer pipeline
    video_capture_settings = config.get('video_capture_settings', '')
    cap = cv2.VideoCapture(video_capture_settings, cv2.CAP_GSTREAMER)

    # Check if camera opened successfully
    if not cap.isOpened():
        logger.error("Failed to open camera.")
        exit()

    # Start time
    start_time = time.time()
    elapsed_time = 0

    # Loop to display the camera feed for 5 seconds to get into position
    while elapsed_time < 5:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to capture image.")
            break

        cv2.imshow('Camera Feed', frame)
        cv2.waitKey(1)

        # Update the elapsed time
        elapsed_time = time.time() - start_time

    # Capture a frame after 5 seconds
    ret, frame = cap.read()
    if ret:
        # Save the captured image
        cv2.imwrite("captured_image.jpg", frame)
        logger.info("Image captured and saved as captured_image.jpg")
    else:
        logger.error("Failed to capture image.")
        exit()

    cap.release()
    cv2.destroyAllWindows()

def extract_character():
    """
    Extracts and processes the character from the captured image.
    """
    logger.info("Extracting character from captured image.")
    # Load the captured image
    image = cv2.imread('captured_image.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to obtain a binary image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Init return values
    x, y, w, h = 0, 0, 0, 0

    # Crop the image around the character detected
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        logger.info(f"Detected character at (x: {x}, y: {y}) with width: {w} and height: {h}")

        cropped_image = image[y:y+h, x:x+w]

        # Downscale the cropped image
        width = 40  # width of the training data
        height = 60  # height of the training data
        downscaled_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_AREA)

        # Display and save the downscaled image
        cv2.imshow('Downscaled Image', downscaled_image)
        cv2.imwrite('captured_image.png', downscaled_image)
        logger.info("Character extracted and saved as captured_image.png")
    else:
        logger.warning("No character detected")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return x, y, w, h

def perform_cuda_inference(config):
    """
    Performs inference using the CUDA library to classify the character.
    """
    logger.info("Starting CUDA inference")
    
    # Load CUDA program
    try:
        cuda_lib = ctypes.CDLL('./libcuda_inference.so')
    except OSError as e:
        logger.error(f"Error loading CUDA library: {e}")
        exit()

    # Load the CUDA functions
    cuda_initiate = cuda_lib.initiate
    cuda_inference = cuda_lib.main_driver
    cuda_shutdown = cuda_lib.shutdown
    
    # Extract configuration data
    weights_files = config.get('weights_layers', [])
    biases_files = config.get('biases_layers', [])
    captured_image_filename = config.get('captured_image_filename', '')
    
    # Convert filenames to bytes for ctypes
    weights_args = [filename.encode() for filename in weights_files]
    biases_args = [filename.encode() for filename in biases_files]
    captured_image_filename = captured_image_filename.encode()
    
    if not weights_args or not biases_args or not captured_image_filename:
        logger.error("Missing information in the configuration file")
        exit()
    
    # Define the argument types for the init function
    cuda_initiate.argtypes = [ctypes.c_char_p] * ((len(weights_args) * 2) + 1) 

    # Define the return type of the main function
    cuda_inference.restype = ctypes.c_int

    # Init the program resources
    logger.info("Allocating program resources")
    cuda_initiate(captured_image_filename, *weights_args, *biases_args)

    # Call the main function and get detected character index
    logger.info("Performing character inference")
    character_index = cuda_inference()
    
    # Release the program resources
    logger.info("Freeing program resources")
    cuda_shutdown()

    logger.info(f"CUDA inference completed with character index: {character_index}")
    return character_index

def main(config):
    """
    Main driver function to execute the entire process.
    """
    logger.info("Program started")
    
    # Open camera and capture image
    capture_image(config)

    # Extract character from captured image and save coordinates
    x, y, w, h = extract_character()
    
    # Get detected character index
    character_index = perform_cuda_inference(config)

    # Load the images
    image_path = 'captured_image.jpg'
    overlay_path = 'letters_png/' + get_image_filename(character_index)

    # Load the captured image
    image = cv2.imread(image_path)

    # Load the overlay image (character picture)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Draw a rectangle around the detected character
    if x and y and w and h:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        logger.info(f'Object at ({x}, {y}) - Width: {w} pixels, Height: {h} pixels')

        # Overlay the PNG image next to the bounding box
        image_result = cvzone.overlayPNG(image, overlay, [x + w, y])
    else:
        logger.warning("Failed to detect character")
        image_result = image  # Display original image if no character detected

    # Display the result
    cv2.imshow('Image', image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    logger.info("Program finished")

if __name__ == '__main__':
    # Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="Character recognition using CUDA.")
    
    # Add argument for config file path
    parser.add_argument('--config', type=str, default='inference_config.json', help='Path to the config JSON file (optional)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load program configuration settings from config file
    logger.info(f"Loading JSON configuration file named: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.error(f"Error: Config file '{args.config}' not found.")
        exit()
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON in config file '{args.config}': {e}")
        exit()

    main(config)
