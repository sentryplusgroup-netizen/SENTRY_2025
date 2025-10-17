import cv2
from ultralytics import YOLO
import time

# --- Initialize USB camera ---
cap = cv2.VideoCapture(0)  # 0 = default USB camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)   # Can be adjusted for performance
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- Load YOLOv8 model ---
model = YOLO("deerdetect.pt")

while True:
    start_time = time.time()

    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured.")
        break

    # Run YOLO detection
    results = model.track(frame, persist=True, tracker='botsort.yaml', conf=0.60, iou=0.05)
    annotated_frame = results[0].plot()

    # Calculate FPS
    end_time = time.time()
    fps = 1 / (end_time - start_time)
    text = f'FPS: {fps:.1f}'

    # Draw FPS text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow("USB Camera", annotated_frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
