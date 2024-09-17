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
   ```
   poetry run python main.py
   ```

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

## Evaluation

The model's performance is evaluated using accuracy, confusion matrix, and a classification report. Visualizations of training curves and sample predictions are also generated.

## Contributing

Contributions to this project are welcome. Please feel free to submit a Pull Request.
