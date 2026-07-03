# Emotion Detector CV
A real time emotion recognition application using OpenCV and a custom built Convolutional Neural Network (CNN) in TensorFlow/Keras to identify facial expressions via webcam.

## Project Overview
This project integrates OpenCV for real-time face detection with a TensorFlow/Keras-based Convolutional Neural Network (CNN) to classify emotional states from a live webcam feed. The system provides immediate visual feedback, overlaying the predicted emotion and confidence score directly onto the video stream.

## Key Features
- **Real-time Processing**: Utilizes optimized Haar Cascade classifiers for efficient face tracking.
- **Deep Learning Classification**: A robust CNN architecture trained to distinguish between seven categorical emotional states.
- **Live Visualization**: Dynamic bounding boxes and confidence level overlays.
- **Intuitive Controls**: Simple keyboard commands for application management.

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
