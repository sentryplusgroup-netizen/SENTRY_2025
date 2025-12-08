from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
import time
import serial
import threading
import logging
from ultralytics.utils import LOGGER

# -------------------------------------------
# Silence ultralytics logs
# -------------------------------------------
LOGGER.setLevel(logging.ERROR)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# -------------------------------------------
# Camera config
# -------------------------------------------
CAM_WIDTH = 256
CAM_HEIGHT = 256
CAM_FPS = 10

# -------------------------------------------
# Initialize camera
# -------------------------------------------
cap = cv2.VideoCapture("/dev/video0")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

# -------------------------------------------
# Serial to Arduino
# -------------------------------------------
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
time.sleep(3)
ser.reset_input_buffer()

# -------------------------------------------
# Load YOLO model
# -------------------------------------------
model = YOLO("Sentry_finModel_1_ncnn_model", task="detect")
model.overrides['half'] = True

# -------------------------------------------
# Tracking hyperparameters
# -------------------------------------------
DETECT_CONF = 0.85   # Confidence to START tracking
TRACK_CONF = 0.30    # Confidence to CONTINUE tracking
CONF_BONUS = 0.20    # Confidence boost for locked ID
LOCK_STABLE = 7      # Frames needed to lock onto a target
LOST_GRACE = 8       # Allowed lost frames before drop

current_target_id = None
id_counts = {}
deer_in_view = False
lost_frames = 0

# -------------------------------------------
# Global frame buffers (NO QUEUES!)
# -------------------------------------------
latest_raw_frame = None
latest_raw_lock = threading.Lock()

latest_annotated_frame = None
latest_annotated_lock = threading.Lock()

# -------------------------------------------
# FastAPI app
# -------------------------------------------
app = FastAPI()

# ============================================================
# THREAD 1 – Frame Grabber (always overwrites latest frame)
# ============================================================
def frame_grabber():
    global latest_raw_frame
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        with latest_raw_lock:
            latest_raw_frame = frame


# ============================================================
# THREAD 2 – YOLO + Tracking + Arduino (REAL-TIME)
# ============================================================
def yolo_worker():
    global latest_raw_frame, latest_annotated_frame
    global current_target_id, id_counts, deer_in_view, lost_frames

    prev_time = time.time()

    while True:
        # Get latest camera frame
        with latest_raw_lock:
            frame = latest_raw_frame.copy() if latest_raw_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        loop_start = time.time()

        # Run YOLO
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            iou=0.50,
            classes=[0]
        )

        res = results[0]
        boxes = res.boxes
        annotated = res.plot(boxes=True, masks=False)

        # -------------------------------------------
        # TARGET SELECTION / TRACKING
        # -------------------------------------------
        if boxes is not None and len(boxes) > 0:
            best_conf = 0
            best_id = None

            # Choose the strongest detected deer
            for box in boxes:
                conf = float(box.conf)
                tid = int(box.id.item()) if box.id is not None else None
                if tid is None:
                    continue

                if conf > best_conf:
                    best_conf = conf
                    best_id = tid

            # ----------------------
            # DETECTION PHASE
            # ----------------------
            if current_target_id is None:
                if best_conf >= DETECT_CONF:
                    # Keep stability only for best ID
                    id_counts = {best_id: id_counts.get(best_id, 0) + 1}

                    if id_counts[best_id] >= LOCK_STABLE:
                        current_target_id = best_id
                        deer_in_view = True
                        id_counts.clear()
                        lost_frames = 0
                        try:
                            ser.write(b"Deer detected\n")
                        except:
                            pass
                else:
                    id_counts.clear()

            # ----------------------
            # TRACKING PHASE
            # ----------------------
            else:
                found_target = False

                for box in boxes:
                    tid = int(box.id.item()) if box.id is not None else None
                    if tid != current_target_id:
                        continue

                    conf = float(box.conf)
                    effective_conf = conf + CONF_BONUS

                    if effective_conf >= TRACK_CONF:
                        found_target = True
                        lost_frames = 0

                        # Get center of target
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

                        # Send coordinates to Arduino
                        try:
                            ser.write(f"{cx},{cy}\n".encode())
                        except:
                            pass

                    break

                # Handle lost target with grace
                if not found_target:
                    lost_frames += 1
                    if lost_frames >= LOST_GRACE:
                        deer_in_view = False
                        current_target_id = None
                        lost_frames = 0
                        try:
                            ser.write(b"No deer\n")
                        except:
                            pass

        else:
            # No detections at all
            if deer_in_view:
                lost_frames += 1
                if lost_frames >= LOST_GRACE:
                    deer_in_view = False
                    current_target_id = None
                    lost_frames = 0
                    id_counts.clear()
                    try:
                        ser.write(b"No deer\n")
                    except:
                        pass

        # FPS overlay
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # Save annotated frame for streaming
        with latest_annotated_lock:
            latest_annotated_frame = annotated.copy()


# ============================================================
# THREAD 3 – Streaming (always latest annotated frame)
# ============================================================
def streaming_handler():
    global latest_annotated_frame

    while True:
        with latest_annotated_lock:
            frame = latest_annotated_frame.copy() if latest_annotated_frame is not None else None

        if frame is None:
            time.sleep(0.01)
            continue

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        jpg = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(0.01)


@app.get("/sentry_stream")
async def video_feed():
    return StreamingResponse(streaming_handler(), media_type="multipart/x-mixed-replace; boundary=frame")


# ============================================================
# Start threads
# ============================================================
threading.Thread(target=frame_grabber, daemon=True).start()
threading.Thread(target=yolo_worker, daemon=True).start()

# Run with:
# uvicorn 'SENTRY_FINAL':app --host 0.0.0.0 --port 5000
