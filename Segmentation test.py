import cv2
from ultralytics import YOLO
import time
import serial

# Initialize USB camera 
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

# Initialize serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
time.sleep(3)  # wait for the serial connection to initialize
ser.reset_input_buffer()
print("Serial connection established")

# Load YOLO model
model = YOLO("Sentrymodel_seg1_ncnn_model", task='segment')  

# Define variables for tracking stability
id_counts = {}
STABLE_FRAMES = 10
Confidence = 0.85

deer_In_view = False # Flag to indicate if deer is in view

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("No frame captured")
        break

    # Run YOLO segmentation + tracking 
    results = model.track(frame, persist=True, tracker='botsort.yaml', conf=Confidence, iou=0.50)
    #print(model.task) - for debugging
    annotated_frame = results[0].plot()      # YOLO auto draws masks + boxes

    # Track active IDs
    active_ids = set()

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Deer detected
        if not deer_In_view:
            ser.write(b"Deer detected\n")
            deer_In_view = True

        for box in results[0].boxes:
            conf = float(box.conf)
            track_id = int(box.id.item()) if box.id is not None else None

            if conf >= Confidence and track_id is not None:
                # Grab bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get center of bounding box
                centerX = int((x1 + x2) / 2)
                centerY = int((y1 + y2) / 2)

                # Draw center point on the annotated frame
                cv2.circle(annotated_frame, (centerX, centerY), 5, (0, 255, 0), -1)
                print(f"{centerX}, {centerY}")

                # Send coordinates to Arduino
                message = f"{centerX},{centerY}\n"
                ser.write(message.encode('utf-8'))
                
                active_ids.add(track_id)
                id_counts[track_id] = id_counts.get(track_id, 0) + 1

                if id_counts[track_id] == STABLE_FRAMES:
                    print(f"Segmented target confirmed (ID {track_id})")

    else:
        # No deer detected
        if deer_In_view:
            ser.write(b"No deer\n")
            deer_In_view = False

    # Reset counts if ID disappears 
    for tid in list(id_counts.keys()):
        if tid not in active_ids:
            id_counts[tid] = 0

    # Display FPS 
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Segmentation + Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
