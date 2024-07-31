/**
 * @file cuda_inference.cu
 * @author Antonio Gonzalez
 * @date July 24, 2024
 * @brief This program utilizes GPU acceleration to perform character recognition on video frames.
 * 
 * The program recieves a frame buffer from a live video feed, grayscales and normalizes it,
 * and then performs a feed-forward operation through a neural network to classify the image.
 * The network weights and biases are read from CSV files.
 * 
 * Dependencies: lodepng for PNG decoding, C++ Standard Library, CUDA Toolkit
 * 
 * Usage:
 *     1. Compile the program using a nvcc compiler:
 *         nvcc -o cu_inference cuda_inference.cu lodepng.cpp
 *     2. Run the executable:
 *         ./cu_inference
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>
#include "lodepng.h"

#define THREADS_PER_BLOCK 256
#define ROWS_LAYER1 2400
#define ROWS_LAYER2 512
#define ROWS_LAYER3 512
#define ROWS_LAYER4 256
#define COLS_LAYER1 512
#define COLS_LAYER2 512
#define COLS_LAYER3 256
#define COLS_LAYER4 62
#define NUM_LAYERS 4
#define MAX_NUM_IMAGES 30
#define WIDTH 40
#define HEIGHT 60

// Pointers for device memory 
float* d_weights[NUM_LAYERS];
float* d_biases[NUM_LAYERS];

extern "C" {
	// Struct to hold the results of the inference
	struct Result {
		int max_index[MAX_NUM_IMAGES];
		float percent_certainty[MAX_NUM_IMAGES];
		double time_taken;
	};
	
	
	// Struct to hold the data for each character
	struct Character {
		float* d_input_layer;
		float* d_output_layer;
		float* d_results[NUM_LAYERS - 1];
		cudaStream_t stream;
	};
	// Holds individual character data
	Character characters[MAX_NUM_IMAGES];



	/**
	 * @brief Performs vector-matrix multiplication. 
	 * 
	 * @param weight_matrix 2D array (matrix) with rows representing input neurons and columns representing output neurons. 
	 * @param input_vector 1D array (vector) representing the outputs from the previous layer and inputs for the current layer (neurons).
	 * @param output_vector 1D array (vector) to hold the output from the matrix-vector operation.
	 * @param rows Number of rows in the matrix.
	 * @param cols Number of columns in the matrix.
	 */
	__global__ void matrix_vector_multiplication(const float *weight_matrix, const float *input_vector, float *output_vector, int rows, int cols) {
	    // Calculate the unique thread index that will serve as the index
	    int col = blockIdx.x * blockDim.x + threadIdx.x;
	    
	    // Check to make sure thread index is within the length of the matrix
	    if (col < cols) {
		float sum = 0;
		// Perform matrix vector multiplication
		for (int row = 0; row < rows; row++) {
		    sum += weight_matrix[row * cols + col] * input_vector[row];
		}
		output_vector[col] = sum;
	    }
	}
	
	

	/**
	 * @brief Adds respective bias to each neuron in the given vector.
	 * 
	 * @param data 1D array (vector) representing the neurons to add to the bias. 
	 * @param bias 1D array (vector) representing the biases for each on the neurons in the given layer.
	 * @param size Number of neurons in the layer.
	 */
	__global__ void add_bias(float *data, const float *bias, int size) {
	    // Calculate the unique thread index that will serve as the index
	    int index = blockIdx.x * blockDim.x + threadIdx.x;

	    // Check to make sure thread index is within the length of the vector
	    if (index < size) {
		data[index] += bias[index];
	    }
	}
	
	

	/**
	 * @brief Applies ReLU activation function
	 * 
	 * @param input_vector 1D array (vector) representing the neurons in the layer. 
	 * @param size Number of neurons in the layer.
	 */
	__global__ void relu(float *input_vector, int size) {
	    // Calculate the unique thread index that will serve as the index
	    int index = blockIdx.x * blockDim.x + threadIdx.x;

	    // Check to make sure thread index is within the length of the vector
	    if (index < size) {
		// Apply the ReLU activation function
		input_vector[index] = fmaxf(input_vector[index], 0.0f);
	    }
	}
	
	

	/**
	 * @brief Reads the CSV file and stores the values in a 1D array
	 * 
	 * @param filename Name of CSV file to process.
	 * @return Vector containing CSV data.
	 */
	std::vector<float> read_CSV(const std::string &filename) {
	    // Holds file data
	    std::vector<float> data;
	    std::ifstream file(filename);
	    std::string line, val;

	    // Iterate over each line
	    while (std::getline(file, line)) {
		std::stringstream ss(line);
		// Iterate over each value and process data
		while (std::getline(ss, val, ',')) {
		    data.push_back(std::stof(val));
		}
	    }
	    return data;
	}
	
	

	/**
	 * @brief Converts PNG file from disk to raw pixel data in memory. Used only for still images.
	 * 
	 * @param filename Name of image to process.
	 * @param width Memory address of the image width.
	 * @param height Memory address of the image height.
	 * @return Vector containing raw pixel data stored represented as an unsigned char.
	 */
	std::vector<unsigned char> png2raw(const char* filename, unsigned& width, unsigned& height) {
	    std::vector<unsigned char> image; // used to store the raw pixels

	    // Decode image
	    unsigned error = lodepng::decode(image, width, height, filename);

	    // If there's an error, display it
	    if (error) {
		std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	    }
	    
	    return image;
	}
	
	
	    
	/**
	 * @brief Grayscales, constrast stretches, and normalizes input image.
	 * 
	 * @param image_data Frame's raw pixel data buffer in RGB format.
	 * @param offset Offset to process each detected character separately. 
	 * @return Grayscaled and normalized image.
	 */
	std::vector<float> grayscale_and_normalize(unsigned char* image_data, int offset) {
		std::vector<float> grayscale_image(WIDTH * HEIGHT);

		// Grayscale image
		for (unsigned y = 0; y < HEIGHT; y++) {
		    for (unsigned x = 0; x < WIDTH; x++) {
		    	unsigned index = offset + 3 * (y * WIDTH + x); // calculate index for RGB format
		    	
		        // Grayscale formula
		        float gray = 0.299f * image_data[index] + 0.587f * image_data[index + 1] + 0.114f * image_data[index + 2];
		        grayscale_image[y * WIDTH + x] = gray;
		    }
		}
		
		// Contrast stretch the image then normalize to the range of (0, 1)
		// Pout = (Pin - c)(b - a/d - c) + a
		// a = 0, b = 255, c = min pixel value, d = max pixel value
		// Find min and max
		auto min_max = std::minmax_element(grayscale_image.begin(), grayscale_image.end());
		float c = *min_max.first;
		float d = *min_max.second;
		
		// Contrast stretch the image
		for (unsigned y = 0; y < HEIGHT; y++) {
		    for (unsigned x = 0; x < WIDTH; x++) {
		        
		        // Contrast stretch formula then normalize to the range of (0, 1)
		        grayscale_image[y * WIDTH + x] = ((grayscale_image[y * WIDTH + x] - c) * (255.0f / (d - c))) / 255.0f;
		    }
		}

		return grayscale_image;
	}
	    
	    
	    
	/**
	 * @brief Performs the Softmax activation function to calculate probability distribution. 
	 * 
	 * @param output_layer 1D array (vector) representing the output layer of the neural network.
	 * @return A probablity distribution representing the network's prediction of the given image.
	 */
	std::vector<float> softmax(const std::vector<float>& output_layer) {
	    std::vector<float> probability_distribution(output_layer.size());
	    float max_value = *max_element(output_layer.begin(), output_layer.end()); // find the max value in the layer

	    // Compute the exponential of each element minus the maximum input value
	    float sum = 0.0;
	    for (size_t neuron = 0; neuron < output_layer.size(); neuron++) {
		probability_distribution[neuron] = std::exp(output_layer[neuron] - max_value);
		sum += probability_distribution[neuron];
	    }

	    // Normalize the output to get probabilities
	    for (size_t neuron = 0; neuron < probability_distribution.size(); neuron++) {
		probability_distribution[neuron] /= sum;
	    }

	    return probability_distribution;
	}
	  	
	  	
	  	
    	/**
	 * @brief Allocates all the necessary device memory for program use and reuse during runtime.  
	 * 
	 * @param weights_filenames Array of the weight CSV file names.
	 * @param biases_filenames Array of the bias CSV file names.
	 * @param NUM_IMAGES Maximum number of images the program will process. 30 by default. 
	 */
    	void initiate (const char* weights_filenames[NUM_LAYERS], const char* biases_filenames[NUM_LAYERS], int NUM_IMAGES) {
		// Read CSV files and save to device memory for program use
		for (int i = 0; i < NUM_LAYERS; i++) {
			std::vector<float> h_weight = read_CSV(weights_filenames[i]);
			std::vector<float> h_bias = read_CSV(biases_filenames[i]);
			
			// Allocate device memory for the weights and biases
			cudaMalloc(&d_weights[i], h_weight.size() * sizeof(float));
			cudaMalloc(&d_biases[i], h_bias.size() * sizeof(float)); 
			
			// Copy weight and bias memory from host to device
			cudaMemcpy(d_weights[i], h_weight.data(), h_weight.size() * sizeof(float), cudaMemcpyHostToDevice);
			cudaMemcpy(d_biases[i], h_bias.data(), h_bias.size() * sizeof(float), cudaMemcpyHostToDevice);
		}

		// Allocate memory for characters
		for (int i = 0; i < NUM_IMAGES; i++) {
			cudaMalloc(&characters[i].d_input_layer, ROWS_LAYER1 * sizeof(float));
			cudaMalloc(&characters[i].d_results[0], COLS_LAYER1 * sizeof(float));
			cudaMalloc(&characters[i].d_results[1], COLS_LAYER2 * sizeof(float));
			cudaMalloc(&characters[i].d_results[2], COLS_LAYER3 * sizeof(float));
			cudaMalloc(&characters[i].d_output_layer, COLS_LAYER4 * sizeof(float));
			cudaStreamCreate(&characters[i].stream);
		}
	}



    	/**
	 * @brief Deallocates the device memory. 
	 * 
	 * @param NUM_IMAGES Maximum number of images the program will process. 30 by default. 
	 */
    	void shutdown(int NUM_IMAGES) {
		// Free device memory for each character
	    	for (int i = 0; i < NUM_IMAGES; i++) {
			cudaFree(characters[i].d_input_layer);
			cudaFree(characters[i].d_output_layer);
			cudaFree(characters[i].d_results[0]);
			cudaFree(characters[i].d_results[1]);
			cudaFree(characters[i].d_results[2]);
			cudaStreamDestroy(characters[i].stream);
	    	}
	    	
	    	// Free device memory for weights and biases
	    	for (int i = 0; i < NUM_LAYERS; i++) {
	    		cudaFree(d_weights[i]);
	    		cudaFree(d_biases[i]);
	    	}
	    }
	
	    

    /**
    * @brief Performs feed forward operation and determines detected character.
    *
    * @param res Result struct to pass data back to the main Python script. 
    * @param image_data Frame buffer containing the raw image data for all the detected characters.
    * @param num_images Number of individual images (# of detected characters) in the frame buffer.
    *
    */		
    void main_driver(Result* res, unsigned char* image_data, int num_images) {
        
        // Grayscale / Normalize the captured images and save it into device memory asynchronously
        std::vector<std::vector<float>> h_input_layers(num_images);
        for (int i = 0; i < num_images; i++) {
        	h_input_layers[i] = grayscale_and_normalize(image_data, (i * (WIDTH * HEIGHT * 3)));
        	cudaMemcpyAsync(characters[i].d_input_layer, h_input_layers[i].data(), ROWS_LAYER1 * sizeof(float), cudaMemcpyHostToDevice, characters[i].stream);
        }
        

        // Perform feed forward operation and time the duration
        auto start = std::chrono::high_resolution_clock::now();
        
        // Launch CUDA kernels for each character
        for (int i = 0; i < num_images; i++) {
        	// Layer 1
		int blocks_per_grid = (COLS_LAYER1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		matrix_vector_multiplication<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(d_weights[0], characters[i].d_input_layer, characters[i].d_results[0], ROWS_LAYER1, COLS_LAYER1);
		add_bias<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[0], d_biases[0], COLS_LAYER1);
		relu<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[0], COLS_LAYER1);

		// Layer 2
		blocks_per_grid = (COLS_LAYER2 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
           	matrix_vector_multiplication<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(d_weights[1], characters[i].d_results[0], characters[i].d_results[1], ROWS_LAYER2, COLS_LAYER2);
		add_bias<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[1], d_biases[1], COLS_LAYER2);
		relu<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[1], COLS_LAYER2);
		
		// Layer 3
		blocks_per_grid = (COLS_LAYER3 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		matrix_vector_multiplication<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(d_weights[2], characters[i].d_results[1], characters[i].d_results[2], ROWS_LAYER3, COLS_LAYER3);
		add_bias<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[2], d_biases[2], COLS_LAYER3);
		relu<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_results[2], COLS_LAYER3);
		
		// Layer 4
		blocks_per_grid = (COLS_LAYER4 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
		matrix_vector_multiplication<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(d_weights[3], characters[i].d_results[2], characters[i].d_output_layer, ROWS_LAYER4, COLS_LAYER4);
		add_bias<<<blocks_per_grid, THREADS_PER_BLOCK, 0, characters[i].stream>>>(characters[i].d_output_layer, d_biases[3], COLS_LAYER4);
        }
        
        // Copy the results back for all characters asynchronously
        std::vector<std::vector<float>> h_output_layers(num_images, std::vector<float>(COLS_LAYER4));
        for (int i = 0; i < num_images; i++) {
		cudaMemcpyAsync(h_output_layers[i].data(), characters[i].d_output_layer, COLS_LAYER4 * sizeof(float), cudaMemcpyDeviceToHost, characters[i].stream);
        }

        // Calculate and save result
        for (int i = 0; i < num_images; i++) {
        	// Calculate probability distribution using softmax function and determine detected character 
		std::vector<float> probability_distribution = softmax(h_output_layers[i]);
		auto max_probability = std::max_element(probability_distribution.begin(), probability_distribution.end());
		int max_index = std::distance(probability_distribution.begin(), max_probability);
            	
            	// Use res struct to return data to Python
		res->max_index[i] = max_index; 
		res->percent_certainty[i] = *max_probability * 100.0f; 
        }

        // Record end time
        auto end = std::chrono::high_resolution_clock::now();

        // Calculate the duration in milliseconds
        std::chrono::duration<double, std::milli> duration = end - start;

        // Record time taken
        res->time_taken = duration.count();
    }
}
