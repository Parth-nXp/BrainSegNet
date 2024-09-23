# BrainSegNet: Brain MRI Segmentation with U-Net in PyTorch

This repository implements **BrainSegNet**, a deep learning model based on the U-Net architecture, for segmenting brain MRI images. The model is implemented in **PyTorch**, making it efficient to run on both CPU and GPU for medical image analysis tasks such as brain tumor detection and other segmentation applications.

## Project Structure

The project is divided into five main scripts:

### 1. `dataset.py`
   - **Purpose**: Contains the `BrainMRIDataset` class, responsible for loading and preprocessing brain MRI images and their corresponding masks.
   - **Key Functionality**:
     - `__getitem__()`: Loads an MRI image and its corresponding mask, resizes them to a target size, normalizes the pixel values, and returns them as PyTorch tensors. The image can optionally be augmented using transformations.
     - `__len__()`: Returns the number of samples (image-mask pairs) in the dataset.
       
### 2. `model.py`
   - **Purpose**: Defines the U-Net architecture, which is used for segmenting brain MRI images.
   - **Key Functionality**:
     - The U-Net architecture is defined with an encoder-decoder structure. The encoder progressively downsamples the input using convolutional layers, while the decoder upsamples it back to the original resolution, concatenating corresponding encoder layers via skip connections.
     - Implements `forward()` method to perform the forward pass of the U-Net. This method defines how the data flows through the encoder, the bottleneck (middle layers), and the decoder to produce a segmentation map.


### 3. `train.py`
   - **Purpose**: Contains the `train_model()` function, which manages the model's training loop, including loss calculation, metric tracking, and model optimization.
   - **Key Functionality**:
     - **Training Loop**: Performs forward and backward passes on the training data, computes the loss using the Dice coefficient loss function, and updates the model’s weights using the Adam optimizer.
     - **Metrics Calculation**: During each epoch, the Dice coefficient and IoU (Intersection over Union) are calculated to assess the segmentation accuracy.
     - **Model Checkpointing**: Saves the model's weights whenever the validation loss improves during training.
     - **Learning Rate Scheduling**: Adjusts the learning rate using a scheduler after a fixed number of epochs.

### 4. `evaluate.py`
   - **Purpose**: Contains the `evaluate_model()` function to visualize the performance of the trained U-Net model on the test dataset.
   - **Key Functionality**:
     - **Visualization**: For a random batch of images from the test set, the original MRI image, the ground truth segmentation mask, and the predicted segmentation mask are displayed side by side. This allows qualitative assessment of the model's performance on unseen data.
     - **Inference**: The model is put into evaluation mode `(model.eval())`, and predictions are generated without gradient computation for efficiency.


### 5. `main.py`
   - **Purpose**: Serves as the main entry point for the project, integrating dataset loading, model training, and model evaluation into one cohesive pipeline.
   - **Key Functionality**:
     - **Dataset Preparation**: Loads the dataset and splits it into training and testing sets. Applies necessary transformations like data augmentation for training.
     - **Model Initialization**: Instantiates the U-Net model, optimizer, and learning rate scheduler.
     - **Training and Validation**: Trains the model using the `train_model()` function and tracks the model’s performance on the validation set.
     - **Evaluation**: After training, the model is evaluated using the `evaluate_model()` function, which visualizes predictions on the test set.

The dataset used in this project comes from two key sources:

1. **Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski**  
   _"Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm." Computers in Biology and Medicine, 2019._

2. **Maciej A. Mazurowski, Kal Clark, Nicholas M. Czarnek, Parisa Shamsesfandabadi, Katherine B. Peters, Ashirbani Saha**  
   _"Radiogenomics of lower-grade glioma: algorithmically-assessed tumor shape is associated with tumor genomic subtypes and patient outcomes in a multi-institutional study with The Cancer Genome Atlas data." Journal of Neuro-Oncology, 2017._

This dataset contains brain MR images together with manual FLAIR abnormality segmentation masks. The images were obtained from **The Cancer Imaging Archive (TCIA)** and correspond to 110 patients included in **The Cancer Genome Atlas (TCGA)** lower-grade glioma collection. Each patient has at least one fluid-attenuated inversion recovery (FLAIR) sequence and genomic cluster data available.

- Tumor genomic clusters and patient data are provided in the `data.csv` file.
- For more information on genomic data, refer to the publication _"Comprehensive, Integrative Genomic Analysis of Diffuse Lower-Grade Gliomas"_ and supplementary material available [here](https://www.nejm.org/doi/full/10.1056/NEJMoa1402121).




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

