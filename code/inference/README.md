# Inference

### Test Page
A test page with 11pt Arial font should be used. Individual characters and/or separate words should have at least a tab's worth of space in between to avoid being represented as a whole word. Individual characters of a word should have exactly 1 space in-between each other to be considered part of the same word so that the program can properly detect individual characters. Characters/words should be at least double spaced apart to be detected on separate lines.

**An example test page can be found in the `assets` folder. Or click [Here](https://jira.a2etechnologies.com:8444/projects/AEOCR/repos/jetson-ocr/browse/assets/Example%20OCR%20Test%20Page.docx)**

---
# Running the System
Navigate to the inference folder.
```bash
cd jetson-ocr/code/inference
```

Rebuild project if necessary:
```bash
make
```

If built successfully, a file called `libcuda_inference.so` should be created, at which point, you are ready to run the program. 

The system is designed to be run from the command line with no additional arguments required. Although, command line arguments are available for further customization and testing purposes. For the full list of arguments use the -h, --help argument:
```
Character recognition using CUDA

Usage: perform_inference.py [-h] [-n NUMBER] [-v] [-c CONFIG]
	Press the "Esc" key to gracefully exit the program.

Options:  
	-h | --help           Print this message  
	-n | --number         Set maximum number of characters to process. Number must be greater than 0 and no more than 30. Default number is 30  
	-v | --verbose       Display FPS, inference time, number of images detected, and other info  
	-c | --config         Path to the config JSON file
```

Example:
```
	python3 perform_inference -n 25 -v     Inference on a maximum of 25 characters and include info like FPS, inference time, and number of images detected.

	python3 perform_inference -c new_config.json     Inference on a maximum of 20 characters (default) using the user-created "new_config.json", without displaying addtional infernce information. 
```
**Note:** Press the "Esc" key to gracefully exit the program.

The GUI should appear with the following elements:
* The viewing window showing the live video stream
* Detected objects encapsulated by a green bounding box 
* Detected character and percent certainty overlaid above the detected object

If the "verbose" option was chosen, in the top left of the window:
* Frames Per Second
* Max # of objects
* Detected # of objects
* Inference time

### Output Products
An output log `program.log` should be created once the program concludes that includes a report of key system operations with respective timestamps for debugging purposes or if unexpected errors are encountered. 

---
# File Information

- **cuda_inference.cu**: This file contains the implementation of a GPU-accelerated character recognition system using CUDA. It processes video frames to perform real-time character classification by leveraging neural network operations accelerated on the GPU. Key functionalities include:
	- **Frame Processing**: Receives a frame buffer from a live video feed, grayscales, and normalizes the frames
	- **Neural Network Inference**: Performs a feed-forward operation through a neural network to classify the characters. Weights and biases for the network are read from CSV files
	- **CUDA Kernels**: Implements CUDA kernels for matrix-vector multiplication, bias addition, and ReLU activation function to accelerate the neural network operations
	- **Softmax Activation**: Applies the softmax activation function to the output layer to get a probability distribution for character classification
	- **Memory Management**: Allocates and manages device memory for weights, biases, and intermediate results for each character being processed
	- **Performance Measurement**: Measures the inference time to evaluate performance
- **inference_config.json**: Contains the essential parameters for running the neural network inference and video capture settings for the OCR project. It specifies the file paths for the trained weights and biases of each layer of the neural network, as well as the settings for capturing video from the camera
- **Makefile**: Used to automate the compilation and building process for creating the shared library `libcuda_inference.so`. It leverages the NVIDIA CUDA Compiler (NVCC) to compile CUDA source files
- **lodepng.cpp**: Source code for LodePNG. [GitHub](https://github.com/lvandeve/lodepng)
- **lodepng.h**: Header files for LodePNG. [GitHub](https://github.com/lvandeve/lodepng)
- **perform_inference.py**: This script performs real-time character recognition from video using CUDA for accelerated inference. It processes frames from a camera, extracts detected characters, and classifies them using a neural network model. Key functionalities include:
	- **Logging**: Configures logging to track program execution and errors, saving logs to `program.log`
	- **Character Extraction**: Processes video frames to extract and downscale character images using OpenCV functions
	- **CUDA Inference**: Interfaces with a CUDA library `libcuda_inference.so` to perform character classification on the GPU
	- **Video Capture**: Opens a video stream from a CSI camera using GStreamer pipeline settings provided in the config file
	- **Inference and Visualization**: Performs inference on the detected characters, draws bounding boxes around words, and displays the processed video with overlayed information like FPS, detected objects, and inference time
	- **Configuration Handling**: Loads configuration settings from a JSON file to adjust paths to model weights and biases, and to set video capture parameters
