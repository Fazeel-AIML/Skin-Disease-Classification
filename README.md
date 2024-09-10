# Skin Disease Classification using CNN and TensorFlow

This repository contains a deep learning project for classifying skin diseases, specifically melanoma and seborrheic dermatitis, using Convolutional Neural Networks (CNN) and TensorFlow. The project includes a Django-based front end to interact with the model and display results.

## Project Overview

The goal of this project is to develop a machine learning model that can accurately classify images of skin lesions into two categories: melanoma and seborrheic dermatitis. The model is built using TensorFlow and CNNs, which are well-suited for image classification tasks. A Django web application serves as the front end, allowing users to upload images and receive classification results.

## Features

- **Image Classification**: Uses CNNs and TensorFlow to classify skin lesions into melanoma or seborrheic dermatitis.
- **Django Front End**: Provides a user-friendly interface for uploading images and viewing results.
- **Real-time Prediction**: Delivers fast and accurate classification results.

## Requirements

- `tensorflow`
- `keras`
- `numpy`
- `opencv-python`
- `Django`
- `Pillow`

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/skin-disease-classification.git
    ```

2. **Navigate to the project directory**:
    ```bash
    cd skin-disease-classification
    ```

3. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```

4. **Install the required packages**:
    ```bash
    pip install tensorflow keras numpy opencv-python Django Pillow
    ```

5. **Set up the Django project**:
    ```bash
    cd my_django_project  # Replace with your Django project directory
    python manage.py runserver
    ```

## Usage

1. **Train the Model**:
   - Ensure you have your dataset organized into `melanoma` and `seborrheic` folders.
   - Run the training script to train the CNN model:
     ```bash
     python train_model.py
     ```

2. **Run the Django Application**:
   - Start the Django server:
     ```bash
     python manage.py runserver
     ```
   - Navigate to `http://127.0.0.1:8000/` in your browser to access the web application.

3. **Upload an Image**:
   - Use the web interface to upload an image of a skin lesion.
   - The model will classify the image and display the result.

## Model Architecture

The CNN model consists of multiple convolutional layers followed by pooling layers, with a final dense layer for classification. The model is trained using TensorFlow with the following architecture:

- **Input Layer**: Image input of size (224, 224, 3)
- **Convolutional Layers**: Several convolutional layers with ReLU activations
- **Pooling Layers**: Max pooling to reduce dimensionality
- **Fully Connected Layers**: Dense layers for final classification
- **Output Layer**: Softmax activation for binary classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Demo

![Demo](path/to/your/gif.gif)  # Add your GIF file path here
