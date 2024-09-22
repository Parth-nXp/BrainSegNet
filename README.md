# BrainSegNet: Brain MRI Segmentation with U-Net in PyTorch

This repository implements **BrainSegNet**, a deep learning model based on the U-Net architecture, for segmenting brain MRI images. The model is implemented in **PyTorch**, making it efficient to run on both CPU and GPU for medical image analysis tasks such as brain tumor detection and other segmentation applications.

## Project Structure

The project is divided into five main scripts:

### 1. `dataset.py`
   - **Purpose**: Contains the `BrainMRIDataset` class, responsible for loading and preprocessing brain MRI images and their corresponding masks.
   - **Key Functionality**:
     - `__getitem__()`: Loads, normalizes, and returns the MRI image and mask as PyTorch tensors.
     - `__len__()`: Returns the size of the dataset.

### 2. `model.py`
   - **Purpose**: Contains the U-Net architecture used for segmenting the MRI images.
   - **Key Functionality**:
     - Defines the U-Net architecture with an encoder-decoder structure using convolutional layers, batch normalization, and ReLU activations.
     - Implements `forward()` method to handle the forward pass through the network.

### 3. `train.py`
   - **Purpose**: Contains the `train_model()` function, which manages the training loop.
   - **Key Functionality**:
     - Performs forward and backward passes, computes the loss using binary cross-entropy with logits, and updates model weights using the Adam optimizer.

### 4. `evaluate.py`
   - **Purpose**: Contains the `evaluate_model()` function to visualize the segmentation performance of the trained model.
   - **Key Functionality**:
     - Displays the original MRI, ground truth mask, and predicted mask side by side for qualitative evaluation.

### 5. `main.py`
   - **Purpose**: The main entry point of the project, responsible for integrating dataset loading, model training, and evaluation.
   - **Key Functionality**:
     - Loads the dataset, trains the model, saves the trained weights, and evaluates the model on the test dataset.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/BrainSegNet.git
    cd BrainSegNet
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv brainseg-env
    source brainseg-env/bin/activate  # On Windows use `brainseg-env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Train the model

Run the `main.py` script to start training the U-Net model:
```bash
python main.py
```

This will:
- Load the brain MRI images and masks.
- Train the U-Net model on the training dataset.
- Save the trained model weights (model.pth) in the working directory.
  
### 2. Evaluate the model
After training, evaluate the model by running the same `main.py` script:
```
python main.py
```

The evaluation part of the script will:
- Load the saved model weights.
- Visualize the original image, the ground truth mask, and the predicted mask for comparison.

## Troubleshooting

If you encounter any issues or errors while running the project, please check the following:

- Ensure all dependencies are installed correctly by running `pip install -r requirements.txt`.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- Verify that the dataset paths in `main.py` are correct.

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.

