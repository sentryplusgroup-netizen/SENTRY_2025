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

DETECT_CONF = 0.80
TRACK_CONF = 0.30
LOCK_STABLE_FRAMES = 5
LOST_GRACE_FRAMES = 8  # how many consecutive "lost" frames we tolerate

JPEG_QUEUE_MAXSIZE = 1

jpeg_queue = Queue(maxsize=JPEG_QUEUE_MAXSIZE)
stop_event = Event()      # for safe shutdown

app = FastAPI()

# ================================================================
#                       YOLO WORKER PROCESS
# ================================================================
def yolo_worker(jpeg_queue: Queue, stop_event: Event):
    print("[YOLO] Worker starting...")

    # --- Box buffer (for smoothness) ---
    last_box = None
    last_box_time = 0.0
    BOX_HOLD_TIME = 1.0  # seconds to keep last box/use it during grace

    # ---- CAMERA ----
    cap = cv2.VideoCapture("/dev/video0")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    # ---- SERIAL ----
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
        time.sleep(3)
        ser.reset_input_buffer()
        ser.write_timeout = 0
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
    lost_frames = 0  # for grace period handling

    # ===============================
    #            MAIN LOOP
    # ===============================
    while not stop_event.is_set():
        start = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        try:
            results = model.track(
                frame,
                persist=True,
                tracker="botsort.yaml",
                conf=0.40,
                iou=0.50,
                classes=[0]
            )
        except Exception as e:
            time.sleep(0.05)
            continue
        
        if not results or results[0] is None:
            time.sleep(0.01)
            continue

        annotated = results[0].plot(boxes=True, masks=False)
        boxes = results[0].boxes

        # ======================================================
        #                 DETECTION + TRACKING
        # ======================================================
        has_boxes = boxes is not None and len(boxes) > 0

        if has_boxes:
            best_conf = 0.0
            best_id = None

            # pick strongest detection
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
                        lost_frames = 0
                        id_counts.clear()
                        if ser:
                            try:
                                ser.write(b"Deer detected\n")
                            except Exception:
                                pass
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
                            lost_frames = 0  # we see the target again

                            # -- bounding box & center --
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2

                            # Store last known good box and time
                            last_box = (x1, y1, x2, y2)
                            last_box_time = time.time()

                            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

                            if ser:
                                try:
                                    ser.write(f"{cx},{cy}\n".encode())
                                except Exception:
                                    pass
                        # we break even if conf < TRACK_CONF because this is the correct ID
                        break

                # --- If the current tracked ID was NOT found in this frame ---
                if not found and deer_in_view:
                    # Send immediate "No deer" on first loss
                    if lost_frames == 0:
                        if ser:
                            try:
                                ser.write(b"No deer\n")
                            except Exception:
                                pass
                    
                    lost_frames += 1

                    # Use buffered box within grace period if it's still fresh
                    if (lost_frames < LOST_GRACE_FRAMES and
                        last_box is not None and
                        (time.time() - last_box_time) <= BOX_HOLD_TIME):

                        x1, y1, x2, y2 = last_box
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2

                        # Only draw buffer box if YOLO did NOT detect anything this frame
                        if not has_boxes:
                            cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)


                        if ser:
                            try:
                                ser.write(f"{cx},{cy}\n".encode())
                            except Exception:
                                pass
                    else:
                        # out of grace or no valid buffer → fully lost
                        if lost_frames >= LOST_GRACE_FRAMES:
                            current_id = None
                            deer_in_view = False
                            lost_frames = 0
                            last_box = None
                            id_counts.clear()
                            if ser:
                                try:
                                    ser.write(b"No deer\n")
                                except Exception:
                                    pass

        # ======================================================
        #                    NO DETECTION
        # ======================================================
        else:
            if deer_in_view:
                lost_frames += 1

                # Try to use buffered box during grace
                if (lost_frames < LOST_GRACE_FRAMES and
                    last_box is not None and
                    (time.time() - last_box_time) <= BOX_HOLD_TIME):

                    x1, y1, x2, y2 = last_box
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if not has_boxes:
                        cv2.circle(annotated, (cx, cy), 5, (0, 255, 0), -1)

                    if ser:
                        try:
                            ser.write(f"{cx},{cy}\n".encode())
                        except Exception:
                            pass
                else:
                    # truly lost after grace
                    if lost_frames >= LOST_GRACE_FRAMES:
                        current_id = None
                        deer_in_view = False
                        lost_frames = 0
                        last_box = None
                        id_counts.clear()
                        if ser:
                            try:
                                ser.write(b"No deer\n")
                            except Exception:
                                pass

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

            # Keep queue fresh (no backlog)
            if jpeg_queue.full():
                try:
                    jpeg_queue.get_nowait()
                except Exception:
                    pass

            try:
                jpeg_queue.put_nowait(jpeg)
            except Exception:
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
        try:
            frame = jpeg_queue.get(timeout=2)  # 2-second timeout to detect worker failure
        except:
            # Queue timeout — worker may have crashed, retry
            time.sleep(0.1)
            continue
        
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
    stop_event.set()  # tell worker to stop
    if yolo_process.is_alive():
        yolo_process.terminate()
        yolo_process.join()


atexit.register(cleanup)

print("[MAIN] YOLO worker started.")
print("[MAIN] Run with:")
#print("uvicorn sentry_multiprocess:app --host 0.0.0.0 --port 5000 --no-access-log")
