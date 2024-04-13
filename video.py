import cv2
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog

# Load YOLOv3 configuration file and trained weights
net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", "yolov3_video_4000.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define classes
classes = ['Wearing Helmet', 'Not Wearing Helmet']

# Define colors for bounding boxes
colors = [(0, 255, 0), (0, 0, 255)]  # Green for Wearing Helmet, Red for Not Wearing Helmet

# Create a Tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Open file dialog to select video file
file_path = filedialog.askopenfilename()

# Check if file was selected
if file_path:
    # Open the selected video file
    cap = cv2.VideoCapture(file_path)
else:
    print("No file selected")

# Process video frame by frame
while True:
    ret, img = cap.read()
    if not ret:
        break

    # Resize the image
    img = cv2.resize(img, (1280, 720))
    height, width, _ = img.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the names of the output layers
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)

    # Initialize lists for boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    # Process each output from the model
    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to remove redundant boxes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding boxes and labels
    font = cv2.FONT_HERSHEY_PLAIN
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = round(confidences[i] * 100, 2)
            
            # Set the color based on whether the person is wearing a helmet
            if label == 'Not Wearing Helmet':
                color = (0, 0, 255)  # Red for not wearing helmet
            else:
                color = (0, 255, 0)  # Green for wearing helmet
            
            # Draw the bounding box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence}%", (x, y - 10), font, 2, color, 2)

    # Display the frame
    cv2.imshow('img', img)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
