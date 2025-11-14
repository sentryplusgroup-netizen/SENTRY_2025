import cv2
from ultralytics import YOLO
import time

# --- Initialize USB camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# --- Load YOLO model ---
model = YOLO("Sentrymodel_seg1.pt")  # Load the NCNN converted model

# --- Start tracking loop 
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured.")
        break

    # Run YOLO tracking (not just detection)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml")

    # Draw annotated frame
    annotated_frame = results[0].plot()

    # Compute total FPS
    fps = 1 / (time.time() - start_time)
    text = f'FPS: {fps:.1f}'
    cv2.putText(annotated_frame, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display
    cv2.imshow("USB Camera Tracking", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()
