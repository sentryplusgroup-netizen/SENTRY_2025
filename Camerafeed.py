import cv2
from ultralytics import YOLO

# --- Set up the USB camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# --- Load YOLO model ---
model = YOLO("best_ncnn_model")  # your trained deer model

# --- Detection loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from USB camera.")
        break

    # Run YOLO model with a higher confidence threshold
    results = model.predict(frame, imgsz=320, conf=0.80, task='track')

    # Annotated frame for display
    annotated_frame = frame.copy()
    deer_detected = False

    # Process detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])     # class index
            conf = float(box.conf[0]) # confidence
            if cls == 0 and conf > 0.80:  # class 0 = deer
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
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
