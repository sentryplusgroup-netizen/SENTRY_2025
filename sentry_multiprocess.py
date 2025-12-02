import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import time
import serial
import logging
from multiprocessing import Process, Queue
import queue as pyqueue

# --------------------------
# Global settings
# --------------------------
LOGGER.setLevel(logging.CRITICAL)  # silence Ultralytics spam

TARGET_FPS = 10                    # desired stable FPS
FRAME_INTERVAL = 1.0 / TARGET_FPS

DETECT_CONF = 0.85                 # confidence to START tracking
TRACK_CONF = 0.35                  # confidence to CONTINUE tracking
LOCK_STABLE_FRAMES = 5             # frames to confirm target

JPEG_QUEUE_MAXSIZE = 2             # keep only the latest frames

# Shared JPEG queue between processes
jpeg_queue: Queue = Queue(maxsize=JPEG_QUEUE_MAXSIZE)

app = FastAPI()


# ================================================================
#                   YOLO WORKER PROCESS
# ================================================================
def yolo_worker(jpeg_queue: Queue):
    print("[YOLO] Initializing worker...")

    # ---- Camera ----
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # ---- Serial ----
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
        time.sleep(3)
        ser.reset_input_buffer()
        print("[YOLO] Serial OK")
    except Exception as e:
        print(f"[YOLO] Serial FAIL: {e}")
        ser = None

    # ---- YOLO ----
    print("[YOLO] Loading model...")
    model = YOLO("Sentry_finModel_1_ncnn_model", task="detect")
    model.overrides["half"] = True
    #model.overrides["imgsz"] = 320
    model.overrides["conf"] = 0.40
    model.overrides["device"] = "cpu"

    current_target_id = None
    id_counts = {}
    deer_in_view = False
    fps_ema = None

    while True:
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.40,
            iou=0.50,
            classes=[0]
        )

        annotated_frame = results[0].plot(boxes=True, masks=False)
        boxes = results[0].boxes

        # ---------------------------
        # DETECTION + TRACKING LOGIC
        # ---------------------------
        if boxes is not None and len(boxes) > 0:

            # -------- FIND BEST ID --------
            best_conf = 0
            best_id = None

            for box in boxes:
                conf = float(box.conf)
                tid = int(box.id.item()) if box.id is not None else None
                if tid is None:
                    continue
                if conf > best_conf:
                    best_conf = conf
                    best_id = tid

            # -------- DETECTION PHASE --------
            if current_target_id is None:
                if best_conf >= DETECT_CONF:

                    # keep only this ID
                    keys = [k for k in id_counts if k != best_id]
                    for k in keys:
                        id_counts.pop(k, None)

                    id_counts[best_id] = id_counts.get(best_id, 0) + 1

                    if id_counts[best_id] >= LOCK_STABLE_FRAMES:
                        current_target_id = best_id
                        deer_in_view = True
                        id_counts.clear()
                        if ser:
                            ser.write(b"Deer detected\n")

                else:
                    id_counts.clear()

            # -------- TRACKING PHASE --------
            else:
                found = False

                for box in boxes:
                    tid = int(box.id.item()) if box.id is not None else None
                    if tid == current_target_id:
                        conf = float(box.conf)

                        if conf >= TRACK_CONF:
                            found = True

                            # ---- center calculation ----
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

                            if ser:
                                ser.write(f"{cx},{cy}\n".encode())
                        break

                if not found:
                    current_target_id = None
                    deer_in_view = False
                    if ser:
                        ser.write(b"No deer\n")

        else:
            # ---- NO DETECTIONS ----
            if deer_in_view:
                deer_in_view = False
                current_target_id = None
                id_counts.clear()
                if ser:
                    ser.write(b"No deer\n")

        # ---------------------------
        # Compute FPS (EMA smoothed)
        # ---------------------------
        elapsed = time.time() - loop_start
        instant_fps = 1.0 / elapsed if elapsed > 0 else TARGET_FPS

        if fps_ema is None:
            fps_ema = instant_fps
        else:
            fps_ema = 0.2 * instant_fps + 0.8 * fps_ema

        cv2.putText(
            annotated_frame,
            f"FPS: {fps_ema:.1f}",
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ---------------------------
        # JPEG ENCODE (low quality)
        # ---------------------------
        ok, buffer = cv2.imencode(
            ".jpg",
            annotated_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 70]
        )
        if ok:
            jpeg = buffer.tobytes()
            try:
                if jpeg_queue.full():
                    try:
                        jpeg_queue.get_nowait()
                    except:
                        pass
                jpeg_queue.put_nowait(jpeg)
            except:
                pass

        # ---------------------------
        # HARD FPS LIMITER
        # ---------------------------
        loop_time = time.time() - loop_start
        sleep_needed = FRAME_INTERVAL - loop_time
        if sleep_needed > 0:
            time.sleep(sleep_needed)


# ================================================================
#                   STREAM WORKER (MAIN PROCESS)
# ================================================================
def mjpeg_generator():
    """Stream JPEGs from queue."""
    while True:
        frame = jpeg_queue.get()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@app.get("/sentry_stream")
async def sentry_stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ================================================================
#                   START YOLO PROCESS
# ================================================================
yolo_process = Process(target=yolo_worker, args=(jpeg_queue,), daemon=True)
yolo_process.start()

print("[MAIN] YOLO worker started.")
print("[MAIN] Run with:")
print("uvicorn sentry_stable_multiprocess:app --host 0.0.0.0 --port 5000")
