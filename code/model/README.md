# Model
This directory is organized into two main components:

1. **Model Training**: Located in the `model-training` folder, this section contains a Python notebook `model_training.ipynb` used for training the model. This notebook provides the necessary scripts and code to retrain the model according to your specific needs.
2. **Trained Model**: The `trained-model` folder houses the pretrained weights and biases used by the CUDA inference program. These files are ready for immediate use with the demo program, eliminating the need for any additional model training. However, if you decide to retrain the model using the provided notebook, please be aware that this will overwrite the existing CSV files in this directory.

Feel free to use the pretrained model for quick demonstrations or retrain it as needed to customize the model for your specific use case.

---
# Dataset Preparation
Before training the model, ensure that the dataset is properly prepared:

1. **Extract Dataset**:
    - The raw dataset is provided as a `.zip` file located in the `data` folder. Extract this file to access the image data required for training
2. **Apply Noise**:
    - To enhance the robustness of the model, noise is applied to the dataset. Use the `training_data_noise.py` script found in the `code/utilities` folder. This script processes the images by adding noise, which helps simulate real-world conditions and improve model generalization

For more details and further instructions please see [README - Data](https://jira.a2etechnologies.com:8444/projects/AEOCR/repos/jetson-ocr/browse/data/README.md?until=7a04c5cde9b1568d155fa5f6695637e190a3b5eb&untilPath=data%2FREADME.md)

---
# Model Training
The `model-training.ipynb` notebook is used to train a deep learning model designed for image classification tasks. Here’s a brief overview of the workflow and model architecture:

1. **Data Loading and Visualization**:
    - The dataset is loaded from the `data/dataset_modified` directory. Images are resized to 60x40 pixels and converted to grayscale
    - The notebook visualizes a batch of images to provide a snapshot of the dataset
2. **Data Preprocessing**:
    - Images are normalized to a range of [0, 1] to standardize the input data for the model.
    - The dataset is split into training (70%), validation (20%), and test (10%) sets
3. **Model Architecture**:
    - The model is a fully connected neural network (FCNN) with the following layers:
        - **Input Layer**: Accepts images of shape (60, 40, 1)
        - **Flatten Layer**: Converts the 2D image arrays into 1D vectors
        - **Hidden Layers**: Three dense layers with 512, 512, and 256 units respectively, each using ReLU activation functions
        - **Output Layer**: A dense layer with 62 units and a softmax activation function for multi-class classification
        - **IMPORTANT**: Do not change the number of layers as the CUDA program expects exactly 4 layers in the model
1. **Training**:
    - The model is trained for 150 epochs using the Adam optimizer and sparse categorical crossentropy loss function
    - Training and validation loss, as well as accuracy, are plotted to visualize the performance over epochs
2. **Evaluation**:
    - The model’s performance is assessed using the test set, and a confusion matrix is displayed to show the classification results
3. **Weights and Biases Extraction**:
    - After training, the weights and biases of each layer are extracted and saved as CSV files in the `trained-model` directory
    - These files can be used for inference in the CUDA program without needing to retrain the model
