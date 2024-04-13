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
# Green for Wearing Helmet, Red for Not Wearing Helmet
colors = [(0, 255, 0), (0, 0, 255)]

# Create a Tkinter root window and hide it
root = tk.Tk()
root.withdraw()

# Open file dialog to select image file
file_path = filedialog.askopenfilename()

# Check if file was selected
if file_path:
    # Read the image file
    img = cv2.imread(file_path)
    # Resize the image if needed
    img = cv2.resize(img, (1280, 720))
else:
    print("No file selected")

height, width, _ = img.shape
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
        if confidence > 0.7:
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

# Draw bounding boxes and labels on the image
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        # Set color based on the class (green for Wearing Helmet, red for Not Wearing Helmet)
        if label == 'Not Wearing Helmet':
            color = (0, 0, 255)  # Red for not wearing helmet
        else:
            color = (0, 255, 0)  # Green for wearing helmet
        
        # Draw bounding box and label
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# Show the image with bounding boxes and labels
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
