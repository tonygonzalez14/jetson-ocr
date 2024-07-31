"""
Script to perform real-time Optical Character Recognition (OCR) using CUDA on the NVIDIA Jetson Orin Nano Development Platform.
The system uses OpenCV to manage live video streaming, detect character-objects, produce an overlay, and interface with the CUDA Inference API.
Efficient data transfer and communication between the Python controller and the GPU-accelerated neural network are facilitated by the CTypes library.

Directory structure:
- This script should be placed in jetson-ocr/code/inference.
- The configuration file should be in jetson-ocr/code/inference.
- The shared object file for CUDA inference should be in jetson-ocr/code/inference.

Dependencies:
- numpy
- opencv-python
- ctypes
- argparse
- json

Usage:
1. Ensure the configuration file (inference_config.json) and the shared object file (libcuda_inference.so) are in the correct directory.
2. Run this script from the command line with optional arguments:
   python3 main_inference.py [-n NUMBER] [-v] [-c CONFIG]

Arguments:
- -n, --number: Set the maximum number of characters to process (default: 30, must be between 1 and 30).
- -v, --verbose: Display additional information such as FPS, inference time, and number of detected objects.
- -c, --config: Path to the configuration JSON file (default: inference_config.json).

The script captures video frames, extracts characters, performs CUDA-based inference, and displays the results with bounding boxes and confidence scores.
"""
import cv2
import time
import ctypes
import logging
import argparse
import json
import numpy as np
import os

NUM_LAYERS = 4
MAX_NUM_IMAGES = 30

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

# Disable propagation to the root logger (prevents logs from being output to console)
logger.propagate = False

# Class to hold the results from the CUDA program
class Result(ctypes.Structure):
	_fields_ = [("max_index", ctypes.c_int * MAX_NUM_IMAGES),
		     ("percent_certainty", ctypes.c_float * MAX_NUM_IMAGES),
		     ("time_taken", ctypes.c_double)]

        	
        	
# Returns the letter of the classified character index
def get_letter(character_index):
   	# Match the given character index to its letter for the overlay
	if 0 <= character_index <= 9:
		return character_index
	elif 10 <= character_index <= 35: # calculate the ASCII value of the corresponding uppercase letter
		letter = chr(ord('A') + (character_index - 10))
		return letter
	elif 36 <= character_index <= 61: # calculate the ASCII value of the corresponding lowercase letter
		letter = chr(ord('a') + (character_index - 36))
		return letter
	else:
		logger.error("Invalid character number")
		return "Invalid character number"
   
	

# Extracts the detected characters from the current frame
def extract_characters(image):
	logger.info("Extracting character(s) from current frame")
	
	# Convert the image to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Apply Gaussian blur to reduce noise
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	# Apply adaptive thresholding to obtain a binary image
	thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

	# Find contours
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	# Sort contours from top to bottom
	contours = sorted(contours, key=lambda c: (cv2.boundingRect(c)[1]))
	
	characters = []
	coordinates = []
	chars_per_line = []
	
	# Loop through contours to extract bounding boxes
	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		if w > 10 and h > 30: # filter out small noise 
			logger.info(f"Detected character at (x: {x}, y: {y}) with width: {w} and height: {h}")
			coordinates.append((x, y, w, h))
			
	# Sort the coordinates in top to bottom, left to right order
	# Check if there is at least 1 character
	if len(coordinates) > 0:
		# Start with the first character's coordinates
		list_of_line_coords = []
		current_line_coords = [coordinates[0]]
		char_counter = 1
		
		# Check if there is more than 1 character
		if len(coordinates) > 1:
			# Seperate each line of characters into their own list starting from the second element
			for index, coord in enumerate(coordinates[1:], start=1):
				# Continue building line if the difference in height between characters does not exceed threshhold
				if coord[1] - coordinates[index - 1][1] < 100:
					current_line_coords.append(coord)
					char_counter += 1 # count how many characters per line
				# Save previous line and start new line if threshhold is exceeded
				# Keep track of number of characters per line 
				else:
					list_of_line_coords.append(current_line_coords)
					current_line_coords = [coordinates[index]]
					chars_per_line.append(char_counter)
					char_counter = 1
		
		# Save last line 
		list_of_line_coords.append(current_line_coords)
		chars_per_line.append(char_counter)
		
		# Sort each line from left to right based on their x-coordinates
		for line in list_of_line_coords:
			line.sort(key=lambda x: x[0])
								
		# Flatten the 2D coordinates list in a 1D array to return
		coordinates = [coord for line in list_of_line_coords for coord in line]
		
		# Loop through coordinates and crop images from frame
		for coord in coordinates:
			# Add 10 pixels of whitespace and ensure cropping coordinates stay within image boundaries
			x_start = max(coord[0] - 10, 0)
			y_start = max(coord[1] - 10, 0)
			x_end = min(coord[0] + coord[2] + 10, image.shape[1])
			y_end = min(coord[1] + coord[3] + 10, image.shape[0])
			
			cropped_image = image[y_start:y_end, x_start:x_end]
			
			# Downscale the cropped image
			width = 40  # width of the training data
			height = 60  # height of the training data
			downscaled_image = cv2.resize(cropped_image, (width, height), interpolation=cv2.INTER_AREA)
			characters.append(downscaled_image)
		
	return characters, coordinates, chars_per_line	



# Performs inference using CTypes to call CUDA program with live video on multiple characters
def perform_cuda_inference(config, verbose):
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

	# Get filenames and path from config file
	weights_files = config.get('weights_layers', [])
	biases_files = config.get('biases_layers', [])
	cwd = os.getcwd().replace("/inference", "") 

	# Append user's path to trained model to filenames 
	weights_files = [cwd + filename for filename in weights_files]
	biases_files = [cwd + filename for filename in biases_files]
	
	# Encode filenames to bytes for ctypes
	weights_args = [filename.encode() for filename in weights_files]
	biases_args = [filename.encode() for filename in biases_files]
	
	# Convert lists to ctypes arrays
	weights_filenames_type = ctypes.c_char_p * NUM_LAYERS
	biases_filenames_type = ctypes.c_char_p * NUM_LAYERS
	weights_filenames = weights_filenames_type(*weights_args)
	biases_filenames = biases_filenames_type(*biases_args)

	# Define the argument types for the functions
	cuda_initiate.argtypes = [weights_filenames_type, biases_filenames_type, ctypes.c_int]
	cuda_inference.argtypes = [ctypes.POINTER(Result), ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
	cuda_shutdown.argtypes = [ctypes.c_int]

	# Init the program resources 
	logger.info("Allocating program resources")
	cuda_initiate(weights_filenames, biases_filenames, MAX_NUM_IMAGES)

	# Construct GStreamer pipeline string for Jetson CSI camera
	pipeline = config.get('video_capture_settings')

	# Open the camera
	cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

	if not cap.isOpened():
		print("Failed to open camera.")
		return
		
	# Init variables for FPS calculation
	start_time = time.time()
	frame_count = 0
	fps = 0

	# Keep camera open until ESC key is pressed
	while True:
		ret, frame = cap.read()
		if not ret:
			print("Failed to capture frame from camera.")
			break

		# Calculate FPS every second
		frame_count += 1
		if frame_count % 10 == 0:
			end_time = time.time()
			elapsed_time = end_time - start_time
			fps = frame_count / elapsed_time
			start_time = end_time
			frame_count = 0
			
		# Extract character from captured image and save coordinates
		characters, coordinates, chars_per_line = extract_characters(frame)
	
		# Display FPS and details on the frame
		logger.info("Displaying overlay on video")
		if verbose:
			cv2.putText(frame, f'FPS: {fps:.2f}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(frame, 'Inference mode: Video', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			cv2.putText(frame, f'Max # of objects: {MAX_NUM_IMAGES}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
			
		# Check if there are between 0-MAX characters
		if len(characters) > 0 and len(characters) <= MAX_NUM_IMAGES:
			# Concatenate the flattened images into a single buffer
			images_buffer = np.concatenate([char.flatten() for char in characters])
		
			# Convert images buffer to ctypes pointer
			character_data = images_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
			
			# Call the main CUDA function and get detected character index
			logger.info("Performing character inference")
			result = Result()
			num_images = len(characters)
			cuda_inference(ctypes.byref(result), character_data, num_images)
			logger.info("CUDA inference completed")
			
			# Keep track of the words on each line
			lines_to_words_dict = {}
			offset = 0
			
			# Process each line in the frame
			for line in range(len(chars_per_line)):
				# Keep track of the coordinates for each of the words
				words_to_coord_dict = {}
				
				# Holds word data
				word = ''
				word_coordinates = []
				
				# Compare x-coordinates between characters of the same line to identify words 
				# Start word with the first character of line
				word = str(get_letter(result.max_index[offset]))
				word_coordinates.append(coordinates[offset])
				
				# Check if there is more than one character detected on the current line
				if chars_per_line[line] > 1:
					# Loop from the second character to last character on the current line and keep track of the index
					for index, coord in enumerate(coordinates[offset + 1:chars_per_line[line] + offset], start=1):
						# If the horizontal distance between the current character and the previous character is less than 30 pixels
						# Then continue building the word and keep track of coordinates
						# If x - (x[index - 1] + w) < 30
						if coord[0] - (coordinates[offset + index - 1][0] + coordinates[offset + index - 1][2]) < 40:
							word += str(get_letter(result.max_index[offset + index]))
							word_coordinates.append(coord)
						else: # Start of a new word
							# Update words_to_coord_dict with previous word and start new word
							words_to_coord_dict[word] = word_coordinates
							word = str(get_letter(result.max_index[offset + index]))
							word_coordinates = [coord]
							
				# Save the word and coordinates of the last processed element on the current line
				words_to_coord_dict[word] = word_coordinates	
				
				# Save the current line's data in the lines_to_words_dict
				lines_to_words_dict[line] = words_to_coord_dict
				
				# Set offset to the index of the first element of the next line to be processed 
				offset += chars_per_line[line]
			
			char_counter = 0
			# Iterate over each line the lines_to_words_dict and draw bounding boxes around each word
			for line in lines_to_words_dict:
				# Iterate over each word in the words_to_coord_dict
				for word in lines_to_words_dict[line]:
					# Get the coordinates of the first and last character of the word for the bounding box
					first_char_x = lines_to_words_dict[line][word][0][0]
					first_char_y = lines_to_words_dict[line][word][0][1]
					first_char_h = lines_to_words_dict[line][word][0][3]
					last_char_x = lines_to_words_dict[line][word][len(word) - 1][0]
					last_char_y = lines_to_words_dict[line][word][len(word) - 1][1]
					last_char_w = lines_to_words_dict[line][word][len(word) - 1][2]
					last_char_h = lines_to_words_dict[line][word][len(word) - 1][3]
					
					
					# Calculate average percent certainty for word
					average_pc = 0
					for _ in range(len(word)):
						average_pc += result.percent_certainty[char_counter]
						char_counter += 1
					average_pc /= len(word)
					
					# Draw bounding box, display word, and % certainty
					cv2.rectangle(frame, (first_char_x - 10, first_char_y - 10), (last_char_x + last_char_w + 10, last_char_y + last_char_h + 10), (0, 255, 0), 2)
					cv2.putText(frame, word, (first_char_x, first_char_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)
					cv2.putText(frame, f'{average_pc:.4}%', (first_char_x, first_char_y - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
					
			# Display additional program information on overlay
			if verbose:
				cv2.putText(frame, f'Detected # of objects: {len(characters)}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(frame, f'Inference time: {result.time_taken:.2f}ms', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

		elif len(characters) > MAX_NUM_IMAGES: # over MAX_NUM_IMAGES objects in frame
			if verbose:
				cv2.putText(frame, f'Detected # of objects: {len(characters)}', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
				cv2.putText(frame, 'Too many detected objects in frame', (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
		else: # no character detected
			if verbose:
				cv2.putText(frame, 'No object detected', (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
	
		# Display the processed frame
		cv2.imshow('CSI Camera Feed', frame)

		# Exit on ESC key
		key = cv2.waitKey(1) & 0xFF
		if key == 27:  # ESC key
			break

	# Release the camera and close all OpenCV windows
	cap.release()
	cv2.destroyAllWindows()
	
	# Release the program reasources
	logger.info("Freeing program resources")
	cuda_shutdown(MAX_NUM_IMAGES)
    	

	
# Main driver
def main(args):
	logger.info("Program started")

	# Load program configuration settings from config file
	logger.info(f"Loading JSON configuration file named: {args.config}")
	try:
		with open(args.config, 'r') as f:
			config = json.load(f)
	except FileNotFoundError:
		print(f"Error: Config file '{args.config}' not found.")
		logger.error(f"Error: Config file '{args.config}' not found.")
		exit()
	except json.JSONDecodeError as e:
		print((f"Error parsing JSON in config file '{args.config}': {e}"))
		logger.error(f"Error parsing JSON in config file '{args.config}': {e}")
		exit()
		
	# Invalid number of images 
	if args.number <= 0 or args.number > 30:
		print("Number of images must be greater than 0 and no more than 30")
		logger.error("Number of images must be greater than 0 and no more than 30")
		exit()
	elif args.number != 30: # user defined max number of characters to process
		logger.info(f"Max number of images set to: {args.number}")
		global MAX_NUM_IMAGES
		MAX_NUM_IMAGES = args.number
		
	logger.info("Program running in video mode")
		
	perform_cuda_inference(config, args.verbose)

	logger.info("Program finished")

if __name__ == '__main__':
	# Initialize ArgumentParser
	parser = argparse.ArgumentParser(description="Character recognition using CUDA.")
	
	# Add argument for max number of characters to process
	parser.add_argument('-n', '--number', type=int, default=30, help='Set maximum number of characters to process. Number must be greater than 1 and no more than 30. Default number is 30')
	
	# Add argument for verbose information display
	parser.add_argument('-v', '--verbose', action='store_true', help='Display FPS, inference time, number of images detected, and other info')
	
	# Add argument for config file path
	parser.add_argument('-c', '--config', type=str, default='inference_config.json', help='Path to the config JSON file')
	
	# Parse arguments
	args = parser.parse_args()

	main(args)
	
