import cv2
from ultralytics import YOLO
import time
import threading
import queue
from ultralytics.utils import LOGGER
import logging
LOGGER.setLevel(logging.CRITICAL)

# --- Frame queue to prevent lag ---
frame_queue = queue.Queue(maxsize=2)  # Keep only latest 2 frames

# --- Load your YOLO model ---
model = YOLO("Sentry_finModel_1_ncnn_model", task="detect")  # replace with your trained model path
model.overrides['half'] = True  # use FP16 for faster inference

# --- Open USB camera (explicit device path) ---
cap = cv2.VideoCapture("/dev/video0")

# If your camera supports MJPEG (most USB cams do), enable it for higher FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# --- Tracking configuration (same as Segmentation test) ---
DETECT_CONF = 0.80   # High accuracy required to START tracking
TRACK_CONF  = 0.35   # Minimum confidence once tracking begins
LOCK_STABLE_FRAMES = 7 # Number of consecutive frames to confirm stable target
current_target_id = None
id_counts = {}  # Dictionary to count stable frames for each ID

deer_In_view = False # Flag to indicate if deer is in view

# Optional: minimum bounding box area to count as a deer
MIN_AREA = 5000  # adjust based on camera distance / resolution

# --- THREAD 1: Frame Reader (captures latest frame, drops old ones) ---
def frame_reader():
    """Read frames continuously, drop old ones to prevent lag"""
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        try:
            frame_queue.put(frame, block=False)  # Drop frame if queue full
        except queue.Full:
            pass  # Discard old frame, keep latest

# --- THREAD 2: YOLO Processing ---
def yolo_loop():
    global current_target_id, id_counts, deer_In_view
    
    fps = 0.0  # For smoothed FPS calculation
    
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        start = time.time()
        annotated_frame = frame.copy()  # Fallback if YOLO fails

        # --- Run YOLO tracking ---
        try:
            results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.40, iou=0.30, classes=[0])
        except Exception as e:
            time.sleep(0.05)
            continue
        
        if not results or results[0] is None:
            time.sleep(0.01)
            continue
        
        annotated_frame = results[0].plot(boxes=True, masks=False)  # Draw boxes only

        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:

            # ------------------------------------
            # PASS 1 — Find the best candidate box
            # ------------------------------------
            best_conf = 0  # Highest confidence found so far
            best_id = None  # Track ID of the deer with highest confidence

            for box in boxes:
                conf = float(box.conf)  # Confidence of this detection
                tid = int(box.id.item()) if box.id is not None else None  # Track ID of this deer
                if tid is None:
                    continue

                if conf > best_conf:  # If this deer is more confident than previous best
                    best_conf = conf  # Update highest confidence
                    best_id = tid  # Update ID of most confident deer

            # ------------------------------------
            # DETECTION PHASE — No target locked
            # ------------------------------------
            if current_target_id is None:
                if best_conf >= DETECT_CONF:

                    # New refinement: Clear counts for any ID that is NOT the current best
                    keys_to_remove = [k for k in id_counts if k != best_id]
                    for k in keys_to_remove:
                        id_counts.pop(k)

                    # Build stability over multiple frames
                    id_counts[best_id] = id_counts.get(best_id, 0) + 1

                    if id_counts[best_id] >= LOCK_STABLE_FRAMES:
                        current_target_id = best_id
                        id_counts.clear()  # Reset counts once locked
                        deer_In_view = True
                        #print(f"[LOCKED] Target acquired — ID {current_target_id}")
                else:
                    # Not confident enough → ignore
                    id_counts.clear()  # Reset counts if not confident
                    continue

            # ------------------------------------
            # TRACKING PHASE — We have a target
            # ------------------------------------
            else:
                found_target = False

                for box in boxes:
                    tid = int(box.id.item()) if box.id is not None else None
                    if tid == current_target_id:

                        conf = float(box.conf)

                        # Low confidence allowed during tracking
                        if conf >= TRACK_CONF:
                            found_target = True

                            # Extract bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                            centerX = (x1 + x2) // 2
                            centerY = (y1 + y2) // 2

                            # Draw center mark
                            cv2.circle(annotated_frame, (centerX, centerY), 5, (0, 255, 0), -1)

                        break

                # ------------------------------------
                # Target LOST
                # ------------------------------------
                if not found_target:
                    #print("[LOST] Target disappeared")
                    deer_In_view = False
                    current_target_id = None

        # ---------------------------
        # No detections at all
        # ---------------------------
        else:
            # No deer detected
            if deer_In_view:
                deer_In_view = False
                current_target_id = None
                id_counts.clear()  # Reset counts if no detections

        # Display tracking status
        # (removed "Deer Detected" text display)

        # --- FPS calculation (smoothed) ---
        measured_fps = 1 / max(1e-6, time.time() - start)
        fps = 0.9 * fps + 0.1 * measured_fps
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Display frame ---
        cv2.imshow("Deer Tracker", annotated_frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Run YOLO loop in background threads
threading.Thread(target=frame_reader, daemon=True).start()
threading.Thread(target=yolo_loop, daemon=True).start()

# Keep main thread alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    pass
finally:
    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
