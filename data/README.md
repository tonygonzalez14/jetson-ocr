
# Training Data

The image training dataset used for this project contains alphanumeric English characters, which include uppercase and lowercase English letters as well as the digits 0-9, totaling 62 classes. In the _dataset_ directory, the images are organized into folders titled 0-9 for digits, *a-z_L* for lowercase letters, and *A-Z_U* for uppercase letters. The images consist of 3475 different font styles for each respective character. The images are represented as 40x60 pixel images in RGB format.

**Note:** The dataset is in its original state with no modifications or noise added. To make the modifications that were used in the training of the model, you must use the _training_data_noise.py_ script (see below).

For more information on training the model please see [README - Model](https://github.com/tonygonzalez14/jetson-ocr/tree/main/code/model)

---
# Dependencies

To run the noise script, you need to have the following Python libraries installed:
* **NumPy**: Version 1.17 or later
* **OpenCV**: Version 4.8 or later

To install the Python dependencies required for this project, follow these steps:

1. **Ensure Python is Installed**: Verify that Python 3.8 or later is installed by running `python --version` or `python3 --version` in your terminal.

2. **Create and Activate a Virtual Environment (Recommended)**:
   - **Linux**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the Required Python Packages**:
   - Ensure you are in the project directory where the `requirements.txt` file is located.
   - Run the following command to install all required packages:
     ```bash
     pip install -r requirements.txt
     ```

4. **Verify Installation**:
   - You can check if the packages were installed correctly by running:
     ```bash
     pip list
     ```
   - This will list all installed packages and their versions.

For detailed instructions on how to use the `requirements.txt` file, see [Python's documentation on virtual environments](https://docs.python.org/3/library/venv.html) and [pip's user guide](https://pip.pypa.io/en/stable/user_guide/).

---
# Extracting the Images

The images are compressed in a zip folder and must be extracted before noise can be added or they can be used for training.

**Step 1:** Navigate to the data directory.
```bash
cd jetson-ocr/data
```

**Step 2:** Extract the images with the tool of your choice.

Once the folder is extracted, you should see all 62 folders and their respective images.

---
# Adding Noise

Once the images have been extracted, they can be used to train the model, but it is recommended to add noise to the images before doing so to replicate the results we achieved.

**Step 1:** Navigate to the utilities directory.
```bash
cd jetson-ocr/code/utilities
```

**Step 2:** Run the noise script.

```bash
python3 training_data_noise.py
```

This process will take 5-10 minutes depending on your system. A new directory named _dataset_modified_ should be created. The dataset is now ready to be used for training.

---
# Additional Information

## Dataset Structure
- _dataset/0-9_: Contains images of digits 0-9.
- _dataset/a-z_L_: Contains images of lowercase letters a-z.
- _dataset/A-Z_U_: Contains images of uppercase letters A-Z.

## Script Information
- _training_data_noise.py_: This script adds synthetic noise to the images to improve the robustness of the trained model. The type and level of noise added can be configured within the script.

## Training Recommendations
- It is advised to use the modified dataset (_dataset_modified_) for training as it helps the model generalize better to real-world noisy data.
- Ensure that your training environment has sufficient resources, as processing the dataset and adding noise can be computationally intensive.

## Troubleshooting
- If you encounter any issues with extracting the images or running the noise script, verify that you have the necessary permissions and that your Python environment is correctly configured.
- Common dependencies for the noise script include libraries such as `numpy` and `opencv-python` make sure these are installed in your environment.
