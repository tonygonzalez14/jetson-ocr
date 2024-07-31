"""
This script captures an image from a camera using the GStreamer pipeline.
It displays the camera feed for 5 seconds to allow for positioning, then captures and saves an image.
Note: This script is no longer in use and was part of the development process.

Dependencies:
- OpenCV

Usage:
1. Ensure you have OpenCV installed.
2. Run the script to capture an image from the camera after a 5-second preview period.
"""

import cv2
import time

# Open the camera with GStreamer pipeline
cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1920, height=1080, framerate=60/1 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink", cv2.CAP_GSTREAMER)

# Check if camera opened successfully
if not cap.isOpened():
    print("Failed to open camera.")
    exit()

# Start time
start_time = time.time()
elapsed_time = 0

# Loop to display the camera feed for 5 seconds to get into position
while elapsed_time < 5:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
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
    print("Image captured and saved as captured_image.jpg")
else:
    print("Failed to capture image.")

cap.release()
cv2.destroyAllWindows()
