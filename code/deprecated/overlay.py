"""
This script performs image classification using a CUDA library and overlays the corresponding
character image onto a captured image. The script:
1. Calls a CUDA program to get the character index.
2. Loads the captured image and the corresponding character image.
3. Processes the captured image to detect the character's bounding box.
4. Overlays the character image onto the detected bounding box.
5. Displays the result.

Note: This script is no longer in use and was part of the development process.

Dependencies:
- OpenCV
- cvzone
- NumPy
- ctypes (for interfacing with the CUDA library)

Usage:
1. Ensure the CUDA library (libcuda_inference.so) is available in the working directory.
2. Place the script in the directory where 'Test Images/captured_image.jpg' and 'letters_png/' are located.
3. Run the script to classify the character and display the result.
"""

import cv2
import cvzone
import numpy as np
import ctypes

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
        return "Invalid character number"

def main():
    # Call CUDA program to determine character
    cuda_lib = ctypes.CDLL('./libcuda_inference.so')
    cuda_lib.main.restype = ctypes.c_int
    character_index = cuda_lib.main()
    print(character_index)

    # Load the images
    image_path = 'Test Images/captured_image.jpg'
    overlay_path = 'letters_png/' + get_image_filename(character_index)

    # Load the captured image
    image = cv2.imread(image_path)

    # Load the overlay image (character picture)
    overlay = cv2.imread(overlay_path, cv2.IMREAD_UNCHANGED)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a rectangle around the outermost contour (around the whole character)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

        # Draw a rectangle around the object
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Print the width and height in pixels
        print(f'Object at ({x}, {y}) - Width: {w} pixels, Height: {h} pixels')

        # Overlay the PNG image next to the bounding box
        image_result = cvzone.overlayPNG(image, overlay, [x - w, y - h])
    else:
        print("Failed to detect image")
        image_result = image  # Set image_result to the original image if detection fails

    # Display the result
    cv2.imshow('Image', image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
