# Dog Breed Classifier

This project is a machine learning application that classifies dog breeds using a deep learning model. It includes a training pipeline, evaluation scripts, and a Streamlit web application for easy interaction with the trained model.

## Live Demo

You can try out the live demo of this application on Streamlit Sharing:

[Dog Breed Classifier App](https://pentochallenge-bauticalla.streamlit.app/)

## Features

- Train a ResNet101 model on a dataset of dog breed images
- Evaluate the model's performance with various metrics
- Visualize training curves and confusion matrix
- Predict dog breeds from uploaded images using a Streamlit web application

## Project Structure

- `app.py`: Streamlit web application for the dog breed classifier
- `main.py`: Main script to run the entire training and evaluation pipeline
- `train.py`: Contains functions for model training
- `evaluate.py`: Functions for model evaluation
- `visualize.py`: Scripts for creating visualizations
- `predict.py`: Standalone script for making predictions on new images
- `model.py`: Definition of the neural network model
- `dataset.py`: Custom dataset classes and data loading functions
- `config.py`: Configuration settings for the project

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/your-username/dog-breed-classifier.git
   cd dog-breed-classifier
   ```

2. Install the required dependencies using Poetry:
   ```
   poetry install
   ```

3. Download the dataset:
   The dataset used for this project can be downloaded from [this Google Drive folder](https://drive.google.com/drive/folders/1kaV8EvPKYssUryEfqV8GMj3WqtOeTM7C?usp=sharing). Place the downloaded `dogs` folder in the root directory of the project.

   **Note:** The dataset used in this project is a slightly modified version of the original. The following changes have been made:
   - Two images have been removed from the original dataset:
     - `poodle/06489f69c03407bbd5ebdeda1d9a1f5b.jpg`
     - `golden_retriever/Golden_Retriever_Hero_2.jpg`
   - The directory `german_shepherd/.comments` has been removed.

   These modifications were made to ensure consistency and remove unnecessary files from the dataset.


## Usage

1. To train the model and generate visualizations:
   
   Without data augmentation:
   ```
   poetry run python main.py
   ```

   With data augmentation:
   ```
   poetry run python main.py --augment
   ```

   The `--augment` flag enables data augmentation during training, which can help improve model performance and generalization.

2. To make predictions on a single image:
   ```
   poetry run python predict.py path/to/your/image.jpg
   ```

3. To run the Streamlit app locally:
   ```
   poetry run streamlit run app.py
   ```

## Model

The project uses a fine-tuned ResNet101 model pretrained on ImageNet. The final fully connected layer is replaced to match the number of dog breed classes in the dataset.

## Data Augmentation

When using the `--augment` flag during training, the following augmentations are applied to the training data:
- Random resized crop (scale: 0.8 to 1.0)
- Random horizontal flip
- Random rotation (up to 15 degrees)
- Random affine transformation (small translations and scaling)

These augmentations help increase the diversity of the training data, potentially improving the model's ability to generalize to new images.

## Model Hosting and Caching

The trained model is hosted on the Hugging Face Hub, making it easily accessible for the Streamlit application. When the Streamlit app runs, it downloads the model from the Hugging Face Hub and caches it using Streamlit's caching mechanism. This approach ensures:

1. Easy deployment: The model doesn't need to be included in the repository.
2. Version control: We can update the model on Hugging Face Hub without changing the application code.
3. Efficient loading: Streamlit's caching prevents unnecessary reloading of the model between sessions.

The model is downloaded and cached in the `load_model()` function in the `app.py` file.

## Evaluation

The model's performance is evaluated using accuracy, confusion matrix, and a classification report. Visualizations of training curves and sample predictions are also generated.

## Generated Images

During the training and evaluation process, the following images are generated:

1. `training_curves.png`: This image shows the training and validation loss curves, as well as the training and validation accuracy curves over the epochs.

2. `confusion_matrix.png`: A heatmap visualization of the confusion matrix, showing the model's performance across different dog breeds.

3. `predictions.png`: Sample predictions made by the model on validation data, displaying both the true labels and predicted labels.

These images can be found in the root directory of the project after running the training script. They provide valuable insights into the model's performance and learning progress.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.
