from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
from ultralytics import YOLO
import time
import serial
import threading
import queue

# Queues for thread communication
frame_queue = queue.Queue(maxsize=3)      # Raw frames from camera
detection_queue = queue.Queue(maxsize=3)  # Results from YOLO
streaming_queue = queue.Queue(maxsize=1)  # Final frames for streaming

# Initialize USB camera 
cap = cv2.VideoCapture("/dev/video0")

# If your camera supports MJPEG (most USB cams do), enable it for higher FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)

# Initialize serial connection to Arduino
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1.0)
time.sleep(3)  # wait for the serial connection to initialize
ser.reset_input_buffer() # reset and clear the buffer on the pi5

# Load YOLO model
model = YOLO("Sentry_finModel_1_ncnn_model", task="detect")  # Load the segmentation model
model.overrides['half'] = True          # use FP16 math - half precision math to save memory and speed up inference 

# FastAPI app for video streaming 
app = FastAPI()

# Global variables for tracking
DETECT_CONF = 0.85   # High accuracy required to START tracking
TRACK_CONF  = 0.35   # Minimum confidence once tracking begins
LOCK_STABLE_FRAMES = 7 # Number of consecutive frames to confirm stable target
current_target_id = None
id_counts = {}  # Dictionary to count stable frames for each ID
deer_In_view = False # Flag to indicate if deer is in view

frame_lock = threading.Lock()
cached_jpeg_frame = None
fps_display = 0

# --- THREAD 1: Frame Grabber ---
def frame_grabber():
    """Continuously read frames from camera (CPU light, GPU idle)"""
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            frame_queue.put(frame, timeout=0.5)
        except queue.Full:
            pass  # Drop frame if buffer full

# --- THREAD 2: YOLO Detection ---
def yolo_detector():
    """Run YOLO inference (GPU busy, waiting for frames is blocked)"""
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # Run detection
        results = model.track(frame, persist=True, tracker='bytetrack.yaml', conf=0.40, iou=0.50, classes=[0])
        annotated_frame = results[0].plot(boxes=True, masks=False)
        
        # Put results in next queue
        try:
            detection_queue.put((frame, annotated_frame, results), timeout=0.5)
        except queue.Full:
            pass

# --- THREAD 3: Tracking & Arduino ---
def tracker_and_serial():
    """Handle tracking logic and Arduino communication"""
    global current_target_id, id_counts, deer_In_view
    
    while True:
        try:
            frame, annotated_frame, results = detection_queue.get(timeout=1)
        except queue.Empty:
            continue
        
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
                        try:
                            ser.write(b"Deer detected\n")
                        except Exception as e:
                            pass
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

                            # Send to Arduino
                            try:
                                msg = f"{centerX},{centerY}\n"
                                ser.write(msg.encode('utf-8'))
                            except Exception as e:
                                pass

                        break

                # ------------------------------------
                # Target LOST
                # ------------------------------------
                if not found_target:
                    try:
                        ser.write(b"No deer\n")
                    except Exception as e:
                        pass
                    deer_In_view = False
                    current_target_id = None

        # ---------------------------
        # No detections at all
        # ---------------------------
        else:
            # No deer detected
            if deer_In_view:
                try:
                    ser.write(b"No deer\n")
                except Exception as e:
                    pass
                deer_In_view = False
                current_target_id = None
                id_counts.clear()  # Reset counts if no detections
        
        # Send to streaming queue
        try:
            streaming_queue.put(annotated_frame, timeout=0.5)
        except queue.Full:
            pass

# --- THREAD 4: Streaming & FPS ---
def streaming_handler():
    """Handle frame encoding and FPS display"""
    global fps_display, cached_jpeg_frame
    prev_time = time.time()
    
    while True:
        try:
            annotated_frame = streaming_queue.get(timeout=1)
        except queue.Empty:
            continue
        
        # Calculate FPS
        current_time = time.time()
        fps_display = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        
        # Add FPS text
        cv2.putText(annotated_frame, f"FPS: {fps_display:.1f}", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode and cache
        try:
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            encoded_frame = buffer.tobytes()
            
            with frame_lock:
                cached_jpeg_frame = encoded_frame
        except Exception as e:
            pass

# FastAPI streaming functions
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

# Start all threads
threading.Thread(target=frame_grabber, daemon=True).start()
threading.Thread(target=yolo_detector, daemon=True).start()
threading.Thread(target=tracker_and_serial, daemon=True).start()
threading.Thread(target=streaming_handler, daemon=True).start()

# Run this in the terminal:
# uvicorn 'Segmentation_test_multithreaded':app --host 0.0.0.0 --port 5000
