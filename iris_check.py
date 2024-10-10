import cv2
import numpy as np
import os

# Sample dictionary with iris images and their labels
iris_images = {
    'PREKSHA': r'F:\CGIproject\prek.jpg',  # save the images in the same project folder
    'DHANU': r'F:\CGIproject\dhanu.jpg',
}

# Initialize ORB detector
orb = cv2.ORB_create()

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to preprocess and extract ORB features from images
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (100, 100))
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

# Load and preprocess images
iris_templates = {}
for label, image_path in iris_images.items():
    iris_templates[label] = preprocess_image(image_path)

# Function to recognize iris from a captured image
def recognize_iris(captured_image, templates):
    captured_image = cv2.resize(captured_image, (100, 100))
    captured_keypoints, captured_descriptors = orb.detectAndCompute(captured_image, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    min_matches = 0
    recognized_label = None

    for label, (keypoints, descriptors) in templates.items():
        if descriptors is not None and captured_descriptors is not None:
            matches = bf.match(descriptors, captured_descriptors)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > min_matches:
                min_matches = len(matches)
                recognized_label = label

    return recognized_label

# Function to capture iris region from the camera feed
def capture_iris_region(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray_frame.shape
    center_x, center_y = width // 2, height // 2
    region_size = 50
    iris_region = gray_frame[center_y - region_size:center_y + region_size, center_x - region_size:center_x + region_size]
    return iris_region

# Function to recognize iris from camera feed
def recognize_iris_from_camera(templates):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces in the frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            recognized_label = "Animal Detected or Object Detected"
        else:
            iris_region = capture_iris_region(frame)
            recognized_label = recognize_iris(iris_region, templates)
            if recognized_label is None:
                recognized_label = "Invalid: No Match"
        
        cv2.putText(frame, f'Recognized: {recognized_label}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        height, width = frame.shape[:2]
        cv2.rectangle(frame, (width // 2 - 50, height // 2 - 50), (width // 2 + 50, height // 2 + 50), (0, 255, 0), 2)
        cv2.imshow('Camera Feed with Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_iris_from_camera(iris_templates)
