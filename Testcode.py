import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Camera setup
picam2 = Picamera2()
picam2.preview_configuration.main.size = (240, 240)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8 custom model
model = YOLO("deer_detect.pt")
CONF_THRESHOLD = 0.5

while True:
    frame = picam2.capture_array()
    results = model(frame, conf=CONF_THRESHOLD)[0]

    boxes = results.boxes
    if boxes is not None and len(boxes) > 0:
        for box, conf, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            if conf < CONF_THRESHOLD:
                continue
            if int(cls) != 0:  # Only deer
                continue

            x1, y1, x2, y2 = map(int, box)
            # Filter out invalid or tiny boxes
            if x2 - x1 <= 1 or y2 - y1 <= 1:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'Deer {conf:.2f}', (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show FPS
    fps = 1000 / results.speed['inference']
    cv2.putText(frame, f'FPS: {fps:.1f}', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
