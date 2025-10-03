import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 320)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8
model = YOLO("deer_detect.onnx")


while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Run YOLO model on the captured frame and store the results
    results = model(frame, imgsz = 320, task='detect')