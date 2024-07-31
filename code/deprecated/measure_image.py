"""
This script processes a captured image by converting it to grayscale, applying adaptive thresholding,
finding contours, and drawing a rectangle around the detected character. It also prints the width
and height of the detected character.
Note: This script is no longer in use and was part of the development process.

Dependencies:
- OpenCV

Usage:
1. Ensure you have the OpenCV library installed.
2. Place the script in the same directory as 'captured_image.jpg'.
3. Run the script to process the image and display the result.
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

# Draw a rectangle around the outermost contour (around the whole character)
if contours:
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Draw a rectangle around the object
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Print the width and height in pixels
    print(f'Object at ({x}, {y}) - Width: {w} pixels, Height: {h} pixels')
else:
    print("Failed to detect image")

# Display the result
cv2.imshow('Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
