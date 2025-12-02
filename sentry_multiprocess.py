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
from multiprocessing import Process, Queue, Event
import atexit

# ==========================
# GLOBAL SETTINGS
# ==========================
LOGGER.setLevel(logging.CRITICAL)

TARGET_FPS = 10
FRAME_INTERVAL = 1.0 / TARGET_FPS

DETECT_CONF = 0.85
TRACK_CONF = 0.35
LOCK_STABLE_FRAMES = 5

JPEG_QUEUE_MAXSIZE = 2

jpeg_queue = Queue(maxsize=JPEG_QUEUE_MAXSIZE)
stop_event = Event()      # <---- NEW for safe shutdown

app = FastAPI()


# ======================================================Event==========
#                       YOLO WORKER PROCESS
# ================================================================
def yolo_worker(jpeg_queue: Queue, stop_event: Event):
    print("[YOLO] Worker starting...")

    # ---- CAMERA ----
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # ---- SERIAL ----
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
        time.sleep(3)
        ser.reset_input_buffer()
        print("[YOLO] Serial OK")
    except Exception as e:
        print(f"[YOLO] Serial FAIL: {e}")
        ser = None

    # ---- YOLO MODEL ----
    print("[YOLO] Loading model...")
    model = YOLO("Sentry_finModel_1_ncnn_model", task="detect")
    model.overrides["conf"] = 0.40
    model.overrides["device"] = "cpu"

    current_id = None
    id_counts = {}
    deer_in_view = False
    fps_ema = None

    # ===============================
    #            MAIN LOOP
    # ===============================
    while not stop_event.is_set():      # <---- UPDATED
        start = time.time()

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

        annotated = results[0].plot(boxes=True, masks=False)
        boxes = results[0].boxes

        # ======================================================
        #                 DETECTION + TRACKING
        # ======================================================
        if boxes is not None and len(boxes) > 0:
            best_conf = 0
            best_id = None

            for box in boxes:
                tid = box.id
                if tid is None:
                    continue
                tid = int(tid.item())

                conf = float(box.conf)
                if conf > best_conf:
                    best_conf = conf
                    best_id = tid

            # ---------------- DETECTION PHASE ----------------
            if current_id is None:
                if best_conf >= DETECT_CONF:
                    id_counts[best_id] = id_counts.get(best_id, 0) + 1

                    if id_counts[best_id] >= LOCK_STABLE_FRAMES:
                        current_id = best_id
                        deer_in_view = True
                        id_counts.clear()
                        if ser:
                            ser.write(b"Deer detected\n")
                else:
                    id_counts.clear()

            # ---------------- TRACKING PHASE ----------------
            else:
                found = False

                for box in boxes:
                    tid = box.id
                    if tid is None:
                        continue
                    tid = int(tid.item())

                    if tid == current_id:
                        conf = float(box.conf)

                        if conf >= TRACK_CONF:
                            found = True

                            # -- center --
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

                            if ser:
                                ser.write(f"{cx},{cy}\n".encode())
                        break

                if not found:
                    current_id = None
                    deer_in_view = False
                    if ser:
                        ser.write(b"No deer\n")

        # ======================================================
        #                    NO DETECTION
        # ======================================================
        else:
            if deer_in_view:
                current_id = None
                deer_in_view = False
                id_counts.clear()
                if ser:
                    ser.write(b"No deer\n")

        # ======================================================
        #                    FPS DISPLAY
        # ======================================================
        elapsed = time.time() - start
        fps_inst = 1.0 / elapsed if elapsed > 0 else TARGET_FPS

        fps_ema = fps_inst if fps_ema is None else (0.2 * fps_inst + 0.8 * fps_ema)

        cv2.putText(
            annotated,
            f"FPS: {fps_ema:.1f}",
            (10, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # ======================================================
        #                    JPEG COMPRESS
        # ======================================================
        ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            jpeg = buffer.tobytes()

            # Keep queue fresh
            if jpeg_queue.full():
                try:
                    jpeg_queue.get_nowait()
                except:
                    pass

            try:
                jpeg_queue.put_nowait(jpeg)
            except:
                pass

        # ======================================================
        #                    FPS LIMITING
        # ======================================================
        sleep_time = FRAME_INTERVAL - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ===============================
    # CLEAN EXIT FOR WORKER
    # ===============================
    print("[YOLO] Worker shutting down...")
    cap.release()
    if ser:
        ser.close()


# ================================================================
#                   STREAMING GENERATOR
# ================================================================
def mjpeg_generator():
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
#                   START WORKER + CLEANUP
# ================================================================
yolo_process = Process(
    target=yolo_worker,
    args=(jpeg_queue, stop_event),
    daemon=False
)
yolo_process.start()


def cleanup():
    print("[MAIN] Cleaning up worker...")
    stop_event.set()              # <---- tell worker to stop
    if yolo_process.is_alive():
        yolo_process.terminate()
        yolo_process.join()


atexit.register(cleanup)

print("[MAIN] YOLO worker started.")
print("[MAIN] Run with:")
print("uvicorn sentry_multiprocess:app --host 0.0.0.0 --port 5000 --no-access-log")
