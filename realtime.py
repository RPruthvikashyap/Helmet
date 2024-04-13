import cv2
import numpy as np

# Load the YOLOv3 model and its configuration file
net = cv2.dnn.readNetFromDarknet("yolov3_custom.cfg", "yolov3_custom_4000.weights")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define the classes
classes = ['Not Wearing Helmet', 'Wearing Helmet']

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        break
    img = cv2.resize(img, (1280, 720))
    height, width, _ = img.shape

    # Create a blob from the image and set it as input for the model
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layers' names and forward the blob through the model
    output_layers_name = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_name)

    # Initialize lists to store boxes, confidences, and class IDs
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            score = detection[5:]
            class_id = np.argmax(score)
            confidence = score[class_id]

            if confidence > 0.5:  # Confidence threshold
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
    font = cv2.FONT_HERSHEY_PLAIN

    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]

            # Set color for bounding boxes
            if label == 'Not Wearing Helmet':
                color = (0, 0, 255)  # Red for not wearing helmet
            elif label == 'Wearing Helmet':
                color = (0, 255, 0)  # Green for wearing helmet

            # Draw the bounding box and label
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), font, 2, color, 2)

    # Display the image
    cv2.imshow('img', img)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
