import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import time
from collections import deque
import numpy as np

# Initialize the Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 320)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLO model
model = YOLO("SentryYOLOv8_1_ncnn_model", task='segment')

# FPS smoothing
fps_history = deque(maxlen=10)

# Stability variables
stable_counter = 0
STABLE_REQUIRED = 1
previous_mask_exists = False

# Filtering thresholds
MIN_AREA = 400
CONF_THRESH = 0.80
SOLIDITY_THRESH = 0.40   # NEW — rejects weird shapes

while True:
    start_time = time.time()

    # Capture frame
    frame = picam2.capture_array()

    # Run YOLO segmentation + tracking
    results = model.track(frame, tracker='bytetrack.yaml', persist=True)

    # Draw YOLO results first
    annotated_frame = results[0].plot()

    # Extract boxes + masks
    boxes = results[0].boxes
    masks = results[0].masks

    mask_valid = False  # default → assume no valid masks

    if masks is not None and boxes is not None:

        # Loop through detected masks
        for i in range(len(masks.xy)):
            polygon = masks.xy[i]
            conf = boxes.conf[i].item()

            # 1️⃣ Confidence filter
            if conf < CONF_THRESH:
                continue

            # 2️⃣ Area filter
            area = cv2.contourArea(polygon)
            if area < MIN_AREA:
                continue

            # 3️⃣ Solidity filter (major fix)
            hull = cv2.convexHull(polygon)
            hull_area = cv2.contourArea(hull)

            if hull_area == 0:
                continue

            solidity = area / hull_area

            if solidity < SOLIDITY_THRESH:
                continue  # skip weird fake shapes

            # If reached here → mask is valid
            mask_valid = True
            break  # no need to check the rest

    # --------------------------
    #   Stability filtering
    # --------------------------
    if mask_valid:
        if previous_mask_exists:
            stable_counter += 1
        else:
            stable_counter = 0
    else:
        stable_counter = 0

    previous_mask_exists = mask_valid

    # If NOT stable enough → hide masks completely
    if stable_counter < STABLE_REQUIRED:
        annotated_frame = frame.copy()  # show raw frame only

    # --------------------------
    #   FPS calculation
    # --------------------------
    frame_time = time.time() - start_time
    fps_history.append(frame_time)
    fps = 1 / (sum(fps_history) / len(fps_history))

    cv2.putText(annotated_frame, f"FPS: {fps:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # Display
    cv2.imshow("Camera", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
