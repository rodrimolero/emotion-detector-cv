import cv2
import os
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model 

# Load trained model
PROJECT_ROOT = Path(__file__).resolve().parent
model = load_model(PROJECT_ROOT / "models" / "emotion_model.keras")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # It will return a list of rectangles where faces are detected wiith the variables:
    #   x: number of pixels from the left of the frame to the left side of the rectangle
    #   y: number of pixels from the top of the frame to the top side of the rectangle
    #   w: width of the rectangle
    #   h: height of the rectangle

    for (x, y, w, h) in faces:
        # Extract the region of interest (the face)
        region_of_interest = gray[y:y+h, x:x+w]
        # Resize the region of interest to 48x48 pixels and normalize the pixel values
        region_of_interest = cv2.resize(region_of_interest, (48, 48)) / 255.0
        # The keras CNN model expects a 4D input (batch_size, height, width, channels), so we 
        #   need to expand the dimensions of the region of interest, producing a final shape of (1, 48, 48, 1)
        region_of_interest = np.expand_dims(region_of_interest, axis=(0, -1))
        
        # Make prediction by passing the face image to the model
        prediction = model.predict(region_of_interest, verbose=0)[0]
        # Get the index of the emotion and the label that corrsponds to it
        emotion_index = np.argmax(prediction)
        label = emotion_labels[emotion_index]

        # Draw the main face box and the predicted emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
        cv2.rectangle(frame, (x - 1, y - 35), (x + w + 1, y), (0, 255, 0), -1)
        cv2.putText(frame, f"{label}: {prediction[emotion_index]*100:.1f}%", (x + 5, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Define the dashboard anchor points
        dash_x = x + w + 10
        dash_y = y

        # Draw the dynamic dashboard
        for i, score in enumerate(prediction):
            # Define the text label and color for the progress bar
            text = f"{emotion_labels[i]} {score*100:.0f}%"
            color = (0, 255, 0) if i == emotion_index else (200, 200, 200) # Green if highest
            
            # Draw the background bar
            cv2.rectangle(frame, (dash_x, dash_y), (dash_x + 120, dash_y + 25), (30, 30, 30), -1)
            # Draw the progress bar
            cv2.rectangle(frame, (dash_x, dash_y), (dash_x + int(score * 120), dash_y + 25), color, -1)
            # Write text
            cv2.putText(frame, text, (dash_x + 5, dash_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            dash_y += 30
    
    # Display the frame with the detected emotions and dashboard
    cv2.imshow('Emotion Detector', frame)

    # Logic to close the frame when the user presses 'q' on their keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()