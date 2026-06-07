# Handwritten Digit Recognizer using CNN

## Overview

This project is a Deep Learning-based Handwritten Digit Recognition System built using a Convolutional Neural Network (CNN) and the MNIST dataset. The model learns to identify handwritten digits from 0–9 with high accuracy and demonstrates the application of Computer Vision and Deep Learning techniques.

---

## Features

* Handwritten digit recognition (0–9)
* Convolutional Neural Network (CNN) architecture
* Automatic dataset preprocessing
* Model evaluation with accuracy and loss metrics
* Confusion Matrix visualization
* Classification Report generation
* Training and validation performance graphs
* Model saving for future inference

---

## Tech Stack

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Dataset

The project uses the MNIST dataset, which contains:

* 60,000 training images
* 10,000 testing images
* Image size: 28 × 28 pixels
* 10 digit classes (0–9)

Dataset is automatically downloaded through TensorFlow/Keras.

---

## CNN Architecture

Input Layer (28×28×1)

↓
Conv2D (32 Filters)
Batch Normalization
MaxPooling2D

↓
Conv2D (64 Filters)
Batch Normalization
MaxPooling2D

↓
Conv2D (128 Filters)
Batch Normalization

↓
Flatten

↓
Dense (256 Neurons)

↓
Dropout (0.4)

↓
Output Layer (10 Classes - Softmax)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/handwritten-digit-recognizer-cnn.git
cd handwritten-digit-recognizer-cnn
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment:

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Required Libraries

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

---

## Run the Project

```bash
python "Digit recognizer.py"
```

---

## Output

The program will:

1. Load and preprocess the MNIST dataset.
2. Train the CNN model.
3. Evaluate model performance.
4. Generate:

   * Accuracy graph
   * Loss graph
   * Confusion matrix
   * Sample predictions
5. Save:

   * digit_recognizer_results.png
   * digit_recognizer_model.h5

---

## Sample Performance

Typical results:

| Metric              | Value |
| ------------------- | ----- |
| Training Accuracy   | ~99%  |
| Validation Accuracy | ~99%  |
| Test Accuracy       | ~99%  |

Results may vary slightly between runs.

---

## Project Structure

Handwritten-Digit-Recognizer-CNN/

│

├── Digit recognizer.py

├── digit_recognizer_model.h5

├── digit_recognizer_results.png

└── README.md

---

## Future Improvements

* Real-time digit prediction using webcam
* Custom image upload support
* Streamlit web application
* Flask deployment
* Mobile application integration

---

Deep Learning | Machine Learning | AI Enthusiast
