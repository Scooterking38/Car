import cv2
import sys

# Load the pre-trained car detection model (Haar Cascade)
car_cascade = cv2.CascadeClassifier('/tmp/haarcascade_car.xml')

# Path to the video file passed as an argument
video_path = sys.argv[1]

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Car counter variable
car_count = 0

while True:
    # Capture frame by frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect cars in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Count the number of cars in the frame
    car_count += len(cars)

# Release the video capture object
cap.release()

# Output the total car count
print(f"Total cars detected: {car_count}")
