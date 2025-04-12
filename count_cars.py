import cv2
import sys
import os
import requests

# URLs for the model files
PROTO_URL = 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/MobileNetSSD_deploy.prototxt'
MODEL_URL = 'https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel'

# Filenames for the model files
PROTO_FILE = 'MobileNetSSD_deploy.prototxt'
MODEL_FILE = 'MobileNetSSD_deploy.caffemodel'

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f'Downloading {filename}...')
        response = requests.get(url)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f'{filename} downloaded successfully.')
        else:
            print(f'Failed to download {filename}. Status code: {response.status_code}')
            sys.exit(1)

# Download the model files if they don't exist
download_file(PROTO_URL, PROTO_FILE)
download_file(MODEL_URL, MODEL_FILE)

# Load the model
net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)

# Class labels from the model
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

# Get the video path from command-line arguments
if len(sys.argv) < 2:
    print("Usage: python count_cars.py <video_path>")
    sys.exit(1)

video_path = sys.argv[1]

# Open the video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    sys.exit(1)

total_cars = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Prepare the frame for detection
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "car":
                total_cars += 1

cap.release()
print(f"Total cars detected: {total_cars}")
