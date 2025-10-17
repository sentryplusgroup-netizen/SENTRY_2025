import cv2
from ultralytics import YOLO
import time

# --- Load your YOLO11 model ---
model = YOLO("yolo11n_custom_ncnn_model")  # replace with your trained model path

# --- Open USB camera ---
cap = cv2.VideoCapture(0)  # try 1 if multiple cameras
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

# --- Detection persistence settings ---
MIN_PERSISTENCE = 3  # minimum consecutive frames to confirm detection
deer_detected_count = 0

# Optional: minimum bounding box area to count as a deer
MIN_AREA = 5000  # adjust based on camera distance / resolution

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured.")
        break

    start = time.time()

    # --- Run YOLO11 tracking ---
    results = model.track(frame, persist=True, tracker='botsort.yaml', conf=0.70, imgsz=320)

    deer_detected_this_frame = False

    if results and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = map(int, xyxy)
            area = (x2 - x1) * (y2 - y1)

            # Only count valid deer detections
            if cls == 0 and conf > 0.70 and area >= MIN_AREA:
                deer_detected_this_frame = True
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Deer {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # --- Update persistence counter ---
    if deer_detected_this_frame:
        deer_detected_count += 1
    else:
        deer_detected_count = 0

    # Confirmed detection only after MIN_PERSISTENCE frames
    if deer_detected_count >= MIN_PERSISTENCE:
        cv2.putText(frame, "🦌 DEER DETECTED!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    # --- FPS calculation ---
    elapsed = time.time() - start
    if 'fps' not in locals():
        fps = 1 / elapsed
    else:
        fps = 0.9 * fps + 0.1 * (1 / elapsed)


    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- Display frame ---
    cv2.imshow("Deer Tracker", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
