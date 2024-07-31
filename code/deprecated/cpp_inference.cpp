/**
 * @file cpp_inference.cpp
 * @author Antonio Gonzalez
 * @date June 10, 2024
 * @brief This program performs inference and predicts the character of a given image.
 * 
 * The program reads a PNG image, converts it to raw pixel data, grayscales and normalizes it,
 * and then performs a feed-forward operation through a neural network to classify the image.
 * The network weights and biases are read from CSV files.
 * Note: This program is no longer in use and was part of the development process.
 * 
 * Dependencies: lodepng for PNG decoding, C++ Standard Library
 * 
 * Usage:
 *     1. Compile the program using a C++ compiler:
 *         g++ -o inference cpp_inference.cpp lodepng.cpp -Wall -Wextra -pedantic -O3
 *     2. Run the executable:
 *         ./inference
 */
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <cmath>
#include "lodepng.h"

/**
 * @brief Converts PNG file from disk to raw pixel data in memory.
 * 
 * @param filename Name of image to process.
 * @param width Memory address of the image width.
 * @param height Memory address of the image height.
 * @return Vector containing raw pixel data stored represented as an unsigned char.
 */
std::vector<unsigned char> png2raw(const char* filename, unsigned& width, unsigned& height) {
    std::vector<unsigned char> image; // used to store the raw pixels

    // Decode
    unsigned error = lodepng::decode(image, width, height, filename);

    // If there's an error, display it
    if (error) {
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
    
    return image;
}

/**
 * @brief Grayscales and normalizes input image.
 * 
 * @param image Image's raw pixel data in RGBA format.
 * @param width Width of image (40px).
 * @param height Height of image (60px).
 * @return Grayscaled and normalized image.
 */
std::vector<float> grayscale_and_normalize(const std::vector<unsigned char>& image, unsigned width, unsigned height) {
    std::vector<float> grayscale_image(width * height);

    // Access every pixel
    for (unsigned y = 0; y < height; y++) {
        for (unsigned x = 0; x < width; x++) {
            unsigned index = 4 * (y * width + x); // Calculate index for RGBA
            float r = static_cast<float>(image[index]);
            float g = static_cast<float>(image[index + 1]);
            float b = static_cast<float>(image[index + 2]);

            // Convert to grayscale
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;

            // Normalize
            grayscale_image[y * width + x] = gray / 255.0f;
        }
    }

    return grayscale_image;
}

/**
 * @brief Reads the given CSV file and stores the values in a 2D array (matrix).
 * 
 * @param filename Name of CSV file to process.
 * @return Vector containing CSV data.
 */
std::vector<std::vector<float>> read_CSV(const std::string &filename) {
    std::ifstream file(filename);
    std::vector<std::vector<float>> data; 
    std::string line;
    
    // Iterate over each row
    while (std::getline(file, line)) {
        std::stringstream line_stream(line);
        std::string cell;
        std::vector<float> row;
        
        // Iterate over each column, ignoring commas 
        while (std::getline(line_stream, cell, ',')) {
            row.push_back(std::stof(cell)); // add column to row
        }
        data.push_back(row); // add row to 2D matrix
    }
    
    return data;
}

/**
 * @brief Performs vector-matrix multiplication.
 * 
 * @param weight_matrix 2D array (matrix) with rows representing input neurons and columns representing output neurons. 
 * @param input_vector 1D array (vector) representing the outputs from the previous layer and inputs for the current layer (neurons).
 * @return 1D array (vector) representing the outputs from the current layer and inputs for the next layer (neurons).
 */
std::vector<float> matrix_vector_multiplication(const std::vector<std::vector<float>>& weight_matrix, const std::vector<float>& input_vector) {
    // Set the size of the output vector (number of columns in a row)
    std::vector<float> output_vector(weight_matrix[0].size(), 0.0f);
    
    // Iterate over each row
    // Each row contains weights for all connections from a specific input neuron to all output neurons
    for (size_t row = 0; row < weight_matrix.size(); row++) {
        // Iterate over each column
        for (size_t col = 0; col < weight_matrix[row].size(); col++) {
            // Calculate weighted sum
            output_vector[col] += weight_matrix[row][col] * input_vector[row];
        }
    }

    return output_vector;
}

/**
 * @brief Adds respective bias to each neuron in the given vector.
 * 
 * @param input_vector 1D array (vector) representing the neurons to add to the bias. 
 * @param bias 1D array (vector) representing the biases for each on the neurons in the given layer.
 * @return 1D array (vector) representing the input vector with the appropriate biases added.
 */
std::vector<float> add_bias(const std::vector<float>& input_vector, const std::vector<float>& biases) {
    std::vector<float> output_vector(input_vector.size());

    // Add respective bias to each neuron 
    for (size_t neuron = 0; neuron < input_vector.size(); neuron++) {
        output_vector[neuron] = input_vector[neuron] + biases[neuron];
    }

    return output_vector;
}

/**
 * @brief Performs the ReLU activation function.
 * 
 * @param weighted_sum The weighted sum of a given neuron.
 * @return The weighted sum if it is greater than zero or zero if it is negative.
 */
float relu(float weighted_sum) {
    return (weighted_sum > 0) ? weighted_sum : 0;
}

/**
 * @brief Performs the Softmax activation function.
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
 * @brief Performs the feed forward operation.
 * 
 * @param grayscaled_image 1D array (vector) representing the grayscaled and normalized pixel data from the image (input layer).
 * @param weights 3D array representing the weight matrices for the layers in the network.
 * @param biases 2D array (matrix) representing the biases for all the neurons in the network.
 * @return 1D array (vector) representing the network's prediction probablity distribution. 
 */
std::vector<float> feed_forward(const std::vector<float>& grayscaled_image, 
                               const std::vector<std::vector<std::vector<float>>>& weights, 
                               const std::vector<std::vector<float>>& biases) {
    
    // Set pixel data as input layer
    std::vector<float> current_output_vector = grayscaled_image;
    
    // Perform feed forward for all layers
    for (size_t layer = 0; layer < weights.size(); layer++) { 
        current_output_vector = matrix_vector_multiplication(weights[layer], current_output_vector);
        current_output_vector = add_bias(current_output_vector, biases[layer]);

        // Apply ReLU function to all the layers except for the last
        if (layer != weights.size() - 1) {
            for (size_t neuron = 0; neuron < current_output_vector.size(); neuron++) {
                current_output_vector[neuron] = relu(current_output_vector[neuron]);
            }
        }
    }

    // Apply softmax activation function to last layer
    return softmax(current_output_vector);
}

int main() {

    // Process CSV files
    std::vector<std::vector<float>> weights1 = read_CSV("weights_layer1.csv");
    std::vector<std::vector<float>> weights2 = read_CSV("weights_layer2.csv");
    std::vector<std::vector<float>> weights3 = read_CSV("weights_layer3.csv");
    std::vector<std::vector<float>> weights4 = read_CSV("weights_layer4.csv");
    std::vector<std::vector<float>> biases1 = read_CSV("biases_layer1.csv");
    std::vector<std::vector<float>> biases2 = read_CSV("biases_layer2.csv");
    std::vector<std::vector<float>> biases3 = read_CSV("biases_layer3.csv");
    std::vector<std::vector<float>> biases4 = read_CSV("biases_layer4.csv");
    
    // Combine weights and biases into vectors of layers
    std::vector<std::vector<std::vector<float>>> weights = {weights1, weights2, weights3, weights4};
    std::vector<std::vector<float>> biases = {biases1[0], biases2[0], biases3[0], biases4[0]};

    // Read in image pixel data
    unsigned width, height;
    std::vector<unsigned char> image = png2raw("test_image_1.png", width, height);

    // Grayscale and normalize the image (input to the network)
    std::vector<float> grayscaled_image = grayscale_and_normalize(image, width, height);
    
    // Record start time
    auto start = std::chrono::high_resolution_clock::now();

    // Perform feed forward
    std::vector<float> output = feed_forward(grayscaled_image, weights, biases);

    // Record end time
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    
    // Print the duration | Average Time: 0.63214 ms
    std::cout << "Time taken: " << duration.count() << " ms" << std::endl;
    
    // Find the max value (highest percentage) in the output layer
    auto max_value = std::max_element(output.begin(), output.end()); 

    // Calculate the index (classified number) of the max value
    int max_index = std::distance(output.begin(), max_value);

    // Print the output 
    std::cout << "Classified character: " << max_index << std::endl;

    // Print the percent certainty
    std::cout << "Percent certainty: " << std::fixed << std::setprecision(2) << *max_value * 100 << "%" << std::endl;
    
    return 0;
}