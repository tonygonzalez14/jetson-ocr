"""
This script processes a captured image to detect and crop characters.
It converts the image to grayscale, applies adaptive thresholding, finds contours,
and crops the image around the detected character. The cropped image is then downscaled
to the desired size. 
Note: This script is no longer in use and was part of the development process.

Dependencies:
- OpenCV

Usage:
1. Ensure you have OpenCV installed.
2. Run the script to process the captured image and display the downscaled result.
"""

import cv2

# Load the captured image
image = cv2.imread('captured_image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to obtain a binary image
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Crop the image around the character detected
if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    cropped_image = image[y:y+h, x:x+w]

    # Downscale the cropped image
    width = 40  # width of the training data
    height = 60  # height of the training data
    downscaled_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_AREA)

    # Display the downscaled image
    cv2.imshow('Downscaled Image', downscaled_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
