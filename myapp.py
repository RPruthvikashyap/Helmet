import streamlit as st
import cv2
import numpy as np
import tempfile

config_path = "yolov3_custom.cfg"
weights_path = "yolov3_custom_4000.weights"
classes = ['Not Wearing Helmet', 'Wearing Helmet']
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
colors = [(0, 0, 255), (0, 255, 0)]

def perform_detection(img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers = net.getUnconnectedOutLayersNames()
    layer_outputs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x, center_y = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = center_x - w // 2, center_y - h // 2
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in np.array(indexes).flatten():
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        confidence = confidences[i]
        color = colors[class_ids[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        label_text = f"{label}: {confidence * 100:.2f}%"
        cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def realtime_detection():
    st.header("🎥 Real-time Detection")
    start_button = st.button("▶️ Start Stream")
    if not start_button:
        st.info("Click 'Start Stream' to begin real-time detection.")
        return
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access webcam. Please check your camera settings.")
        return
    
    video_placeholder = st.empty()
    stop_button = st.button("⏹️ Stop Stream")
    st.text("Streaming... Press 'Stop Stream' to end the stream.")
    
    while True:
        ret, img = cap.read()
        if not ret:
            st.error("Failed to read frame. Stream stopped.")
            break
        
        img = perform_detection(img)
        video_placeholder.image(img, channels="BGR", use_column_width=True)
        
        if stop_button:
            break
    
    cap.release()
    cv2.destroyAllWindows()
    st.text("Stream stopped.")

def video_detection():
    st.header("🎬 Video Detection")
    video_file = st.file_uploader("Upload a video file (mp4, avi, mov, mkv)", type=["mp4", "avi", "mov", "mkv"])
    if video_file is not None:
        st.text("Processing video...")
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(video_file.getvalue())
            temp_file_name = temp_file.name
        
        cap = cv2.VideoCapture(temp_file_name)
        if not cap.isOpened():
            st.error("Failed to open the video file. Please check the file format or try another file.")
            return
        
        video_placeholder = st.empty()
        stop_button = st.button("⏹️ Stop Video")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("End of video or failed to read frame.")
                break
            
            processed_frame = perform_detection(frame)
            video_placeholder.image(processed_frame, channels="BGR", use_column_width=True)
            
            if stop_button:
                break
        
        cap.release()
        st.text("Video processing complete.")

def image_detection():
    st.header("🖼️ Image Detection")
    image_file = st.file_uploader("Upload an image file (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        st.info("Processing image...")
        file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        img = perform_detection(img)
        st.image(img, channels="BGR", use_column_width=True)
        st.success("Image processing complete.")

def main():
    st.sidebar.title("🚦 Helmet Detection App")
    option = st.sidebar.radio("Choose detection type:", ["Real-time Detection", "Video Detection", "Image Detection"])
    
    if option == "Real-time Detection":
        realtime_detection()
    elif option == "Video Detection":
        video_detection()
    elif option == "Image Detection":
        image_detection()

if __name__ == "__main__":
    main()
