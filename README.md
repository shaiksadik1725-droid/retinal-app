# Retinal Disease Classification Web Application

A computer-vision research prototype that classifies retinal images using a trained convolutional neural network and exposes the model through a Flask web application.

## Overview

The project combines deep learning, image preprocessing, model inference, and a browser-based interface. Users upload a retinal image, the application validates the file, runs the trained TensorFlow/Keras model, and returns the predicted class with a confidence score.

## Main Features

- CNN-based retinal image classification
- Flask web interface for image upload and inference
- Support for PNG, JPG, JPEG, BMP, and WebP images
- Saved Keras model and class-index mapping
- Training and model-plot utilities
- Result pages for prediction output
- Health/status endpoint for deployment checks

## Technology Stack

- Python
- TensorFlow / Keras
- Flask
- NumPy
- Pillow
- HTML / CSS
- Gunicorn

## Project Structure

```text
retinal-app/
├── app.py
├── predict.py
├── Train_Model_Plots.py
├── requirements.txt
├── models/
├── templates/
├── static/
├── dataset/
├── Images/
├── Model_Plots_Samples/
└── Model_Results/
```

## How It Works

1. A user uploads a retinal image through the web interface.
2. The image is stored temporarily in the application upload folder.
3. The image is resized to the model input size and normalized.
4. The trained CNN produces class probabilities.
5. The application displays the predicted retinal class and confidence.

## Run Locally

```bash
git clone https://github.com/shaiksadik1725-droid/retinal-app.git
cd retinal-app
pip install -r requirements.txt
python app.py
```

Open:

```text
http://localhost:5000
```

## Research / Educational Use

This repository is an academic and research prototype. Model predictions should not be treated as a medical diagnosis or used as a substitute for professional clinical evaluation.

## Future Improvements

- Add external validation on independent datasets
- Add explainability visualizations such as Grad-CAM
- Improve dataset documentation and reproducibility
- Add automated tests and continuous integration
- Package model configuration and preprocessing more formally

## Author

**Sadik Shaik**

Computer Engineering / AI & Embedded Systems Projects
