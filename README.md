# Emotion Detector CV
A real time emotion recognition application using OpenCV and a custom built Convolutional Neural Network (CNN) in TensorFlow/Keras to identify facial expressions via webcam.

## Project Overview
This project integrates OpenCV for real-time face detection with a TensorFlow/Keras-based Convolutional Neural Network (CNN) to classify emotional states from a live webcam feed. The system provides immediate visual feedback, overlaying the predicted emotion and confidence score directly onto the video stream.

## Technical Architecture & Methodology

### Data & Model
- **Dataset**: The model was trained on the FER-2013 dataset, leveraging its 35,000 images to learn a diverse range of facial expressions.
- **Architecture**: A custom Convolutional Neural Network (CNN) designed for spatial feature extraction. The model utilizes stacked Conv2D layers to identify hierarchical facial geometries, effectively distinguishing between micro-expressions.
- **Pipeline**: Built using TensorFlow/Keras, the pipeline encompasses the entire lifecycle: from data normalization and augmentation to weight optimization and final deployment.

### Real-Time Pipeline
- **Computer Vision**: Employs OpenCV for high-performance face detection. The system captures video frames, applies grayscale conversion, and performs real-time pre-processing to ensure compatibility with the CNN input requirements (48x48 pixels).
- **Inference**: The model executes inference directly on the detected face regions, outputting a probabilistic distribution across the seven emotion classes, rendered visually as a live dashboard.

## Key Features
- **Real-time Processing**: Utilizes optimized Haar Cascade classifiers for efficient face tracking.
- **Robust Classification**: A robust CNN architecture trained to distinguish between seven categorical emotional states.
- **Live Visualization**: Real-time facial bounding boxes with dynamic, per-emotion confidence scores.

## Installation & Setup
1. Clone the repository
```
git clone https://github.com/rodrimolero/emotion-detection-cv.git
cd emotion-detection-cv
```
2. Install dependencies
```
pip install -r requirements.txt
```
3. Model configuration
   Ensure your trained model file (`emotion_model.keras`) is placed within the `/models` directory.

## Usage
Run the application using the following command:
```
python detect_emotions.py
```
- Start: The application will launch a window titled "Emotion Detector" and activate your webcam.
- Exit: Press the 'q' key at any time to close the application.
