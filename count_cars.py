import cv2
import sys
import os
import requests

def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, "wb") as f:
            f.write(response.content)
        print(f"{os.path.basename(path)} downloaded successfully.")
    else:
        print(f"Failed to download {os.path.basename(path)}. Status code: {response.status_code}")
        sys.exit(1)

video_path = sys.argv[1] if len(sys.argv) > 1 else "video.mp4"
print(f"Trying to open: {video_path}")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
    sys.exit(1)

# URLs from djmv repo
PROTOTXT_URL = "https://raw.githubusercontent.com/djmv/MobilNet_SSD_opencv/master/MobileNetSSD_deploy.prototxt"
MODEL_URL = "https://github.com/djmv/MobilNet_SSD_opencv/raw/master/MobileNetSSD_deploy.caffemodel"

PROTOTXT_PATH = "deploy.prototxt"
MODEL_PATH = "MobileNetSSD_deploy.caffemodel"

if not os.path.exists(PROTOTXT_PATH):
    download_file(PROTOTXT_URL, PROTOTXT_PATH)

if not os.path.exists(MODEL_PATH):
    download_file(MODEL_URL, MODEL_PATH)

net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

car_count = 0
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 5 != 0:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    cars_in_frame = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "car":
                cars_in_frame += 1

    if cars_in_frame > 0:
        car_count += cars_in_frame

cap.release()
print(f"Total cars detected: {car_count}")
