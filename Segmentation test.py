import cv2
from ultralytics import YOLO
import time

# --- Initialize USB camera ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)

# --- Load YOLO segmentation model ---
model = YOLO("Seg_yolov8n_model1.pt")   # make sure this is a SEG model, not detection-only

# --- Tracking memory for ID stability ---
id_counts = {}
STABLE_FRAMES = 1
CONF_THRESHOLD = 0.85

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("❌ No frame captured.")
        break

    # --- Run YOLO segmentation + tracking ---
    results = model.track(frame,
                          persist=True,
                          tracker='botsort.yaml',
                          conf=CONF_THRESHOLD,
                          iou=0.50,
                          task="segment")   # <--- IMPORTANT FOR MASKS

    annotated_frame = results[0].plot()      # YOLO auto draws masks + boxes

    # --- Track active IDs ---
    active_ids = set()

    if results[0].boxes is not None:
        for box in results[0].boxes:
            conf = float(box.conf)
            track_id = int(box.id.item()) if box.id is not None else None

            if conf >= CONF_THRESHOLD and track_id is not None:
                active_ids.add(track_id)
                id_counts[track_id] = id_counts.get(track_id, 0) + 1

                if id_counts[track_id] == STABLE_FRAMES:
                    print(f"🦌 Segmented target confirmed internally (ID {track_id})")

    # --- Reset counts if ID disappears ---
    for tid in list(id_counts.keys()):
        if tid not in active_ids:
            id_counts[tid] = 0

    # --- Display FPS ---
    fps = 1 / (time.time() - start_time)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("USB Camera (Segmentation + Tracking)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
