from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
import time
import serial
import threading

# Initialize USB camera 
cap = cv2.VideoCapture("/dev/video7")

# If your camera supports MJPEG (most USB cams do), enable it for higher FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
cap.set(cv2.CAP_PROP_FPS, 30)

# Initialize serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
time.sleep(3)  # wait for the serial connection to initialize
ser.reset_input_buffer() # reset and clear the buffer on the pi5
#print("Serial connection established")

# Load YOLO model
model = YOLO("Sentrymodel_seg1_ncnn_model", task="segment")  # Load the segmentation model
model.overrides['half'] = True          # use FP16 math - half precision math to save memory and speed up inference 

# FastAPI app for video streaming 
app = FastAPI()
latest_frame = None
cached_jpeg_frame = None  # Cache encoded JPEG to avoid re-encoding

def generate_frames():
    global cached_jpeg_frame
    while True:
        if cached_jpeg_frame is None:
            continue
        with frame_lock:
            frame = cached_jpeg_frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get('/sentry_stream')
async def video_feed():
    return StreamingResponse(generate_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')

# Define variables for tracking stability
DETECT_CONF = 0.85   # High accuracy required to START tracking
TRACK_CONF  = 0.35   # Minimum confidence once tracking begins
LOCK_STABLE_FRAMES = 7 # Number of consecutive frames to confirm stable target
current_target_id = None
id_counts = {}  # Dictionary to count stable frames for each ID

deer_In_view = False # Flag to indicate if deer is in view

# Add this at the top with other imports/globals
frame_lock = threading.Lock()

def yolo_loop():
    global latest_frame, cached_jpeg_frame, deer_In_view
    global current_target_id, id_counts

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("No frame captured")
            break

        # Run YOLO segmentation + tracking 
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.40, iou=0.50)
        #print(model.task) - for debugging
        annotated_frame = results[0].plot()      # YOLO auto draws masks and boxes

        # ---------------------------
        # If YOLO detected something
        # ---------------------------
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:

            # ------------------------------------
            # PASS 1 — Find the best candidate box
            # ------------------------------------
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
                        ser.write(b"Deer detected\n")
                        print(f"[LOCKED] Target acquired — ID {current_target_id}")
                else:
                    # Not confident enough → ignore
                    id_counts.clear()  # Reset counts if not confident
                    latest_frame = annotated_frame
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

                            # Send to Arduino
                            msg = f"{centerX},{centerY}\n"
                            ser.write(msg.encode('utf-8'))

                        break

                # ------------------------------------
                # Target LOST
                # ------------------------------------
                if not found_target:
                    #print("[LOST] Target disappeared")
                    ser.write(b"No deer\n")
                    deer_In_view = False
                    current_target_id = None

        # ---------------------------
        # No detections at all
        # ---------------------------
        else:
            # No deer detected
            if deer_In_view:
                ser.write(b"No deer\n")
                deer_In_view = False
                current_target_id = None
                id_counts.clear()  # Reset counts if no detections

        # Display FPS 
        fps = 1 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Encode frame once and cache for streaming (improves performance) ---
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        encoded_frame = buffer.tobytes()
        with frame_lock:
            latest_frame = annotated_frame  # Update latest frame for tracking logic
            cached_jpeg_frame = encoded_frame  # Update cached JPEG for streaming

        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

# Run YOLO loop in background thread
threading.Thread(target=yolo_loop, daemon=True).start()

# Run this in the terminal:
# uvicorn 'Segmentation test':app --host 0.0.0.0 --port 5000
