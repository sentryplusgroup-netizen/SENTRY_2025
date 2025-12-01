import cv2
from ultralytics import YOLO
import time

# --- Initialize USB camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# --- Load YOLO model ---
model = YOLO("Sentry_finModel_1_ncnn_model")

# --- Tracking memory for ID stability ---
id_counts = {}               # Track how many frames each ID has persisted
STABLE_FRAMES = 5            # Confirm after 5 consistent frames
CONF_THRESHOLD = 0.75        # Confidence required

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured.")
        break

    # --- Run YOLO tracking ---
    results = model.track(frame, tracker='bytesort.yaml', conf=CONF_THRESHOLD, iou=0.40)
    annotated_frame = results[0].plot()  # Regular YOLO annotation only

    # --- Track active IDs this frame ---
    active_ids = set()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf)
            track_id = int(box.id.item()) if box.id is not None else None

            # All detections are deer (single-class), just check confidence and valid ID
            if conf >= CONF_THRESHOLD and track_id is not None:
                active_ids.add(track_id)
                id_counts[track_id] = id_counts.get(track_id, 0) + 1

                # Internally you can still detect a "stable ID" for hardware triggers
                # No extra annotation on frame needed
                if id_counts[track_id] == STABLE_FRAMES:
                    print(f"🦌 Deer confirmed internally (ID {track_id})")  # Optional debug

    # --- Reset counters for IDs that disappeared ---
    for tid in list(id_counts.keys()):
        if tid not in active_ids:
            id_counts[tid] = 0

    # --- Display FPS ---
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # --- Show frame ---
    cv2.imshow("USB Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
