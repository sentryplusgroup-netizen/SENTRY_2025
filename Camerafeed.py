import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Set up the camera ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (320, 320)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# --- Load YOLO model ---
model = YOLO("best_ncnn_model")  # your trained deer model

# --- Detection loop ---
while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()

    # Run YOLO model with a higher confidence threshold
    results = model.predict(frame, imgsz=320, conf=0.7, task='detect')

    # Annotated frame for display
    annotated_frame = frame.copy()

    deer_detected = False  # track if any valid deer found

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])           # class index
            conf = float(box.conf[0])       # confidence
            if cls == 0 and conf > 0.7:     # class 0 = deer
                deer_detected = True
                # Draw bounding box
                xyxy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Deer {conf:.2f}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Print detection status
    if deer_detected:
        print("🦌 DEER DETECTED!")

    # --- FPS display ---
    if len(results) > 0:
        inference_time = results[0].speed['inference']
        fps = 1000 / inference_time if inference_time > 0 else 0
        text = f'FPS: {fps:.1f}'
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = annotated_frame.shape[1] - text_size[0] - 10
        text_y = text_size[1] + 10
        cv2.putText(annotated_frame, text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- Display the resulting frame ---
    cv2.imshow("Camera", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) == ord("q"):
        break

# --- Cleanup ---
cv2.destroyAllWindows()
