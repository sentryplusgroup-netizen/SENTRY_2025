import cv2
import time
import numpy as np
import threading
import hailo_sdk_client
# HAILO PACKAGES (from Core Electronics guide)
from hailo_sdk_client import ClientRunner
from hailo_platform import Device, HailoRTException
from hailo_apps_common.hailo_pipeline import create_hailo_device, make_pipeline
from hailo_apps_common.display_utils import show_image

# =============================
# HAILO SETUPZZZZZ
# =============================

# Load your HEF model (Hailo converted YOLO model)
HEF_PATH = "your_model.hef"

try:
    device = create_hailo_device()
    runner = ClientRunner(hef_path=HEF_PATH, device_ids=device.get_device_ids())
    network_group = runner.load()
except HailoRTException as e:
    print("❌ Hailo Error:", e)
    exit()

# =============================
# CALLBACK (where detections happen)
# =============================

latest_frame = None
latest_detections = []

def detection_callback(frame, detections):
    """
    This callback runs on every frame Hailo processes.
    'detections' structure contains: bbox, class_id, confidence, etc.
    """
    global latest_frame, latest_detections
    
    latest_frame = frame
    latest_detections = detections


# =============================
# CREATE THE PIPELINE
# =============================
pipeline = make_pipeline(
    runner,
    input_type="rpi",           # Raspberry Pi camera OR USB (“v4l2”)
    callback=detection_callback
)

# Start the Hailo pipeline (async)
pipeline.start()


# =============================
# MAIN DISPLAY LOOP
# =============================
stable_counter = 0
STABLE_REQUIRED = 3

while True:
    start = time.time()

    if latest_frame is None:
        continue

    frame = latest_frame.copy()

    # Draw detections
    for det in latest_detections:
        x1, y1, x2, y2 = det.bbox
        conf = det.confidence
        cls = det.class_id

        # Stability counter
        if conf > 0.6:
            stable_counter += 1
        else:
            stable_counter = 0

        if stable_counter >= STABLE_REQUIRED:
            # Compute center for turret logic
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            print(f"Stable Detection → cls:{cls} conf:{conf:.2f} center({cx},{cy})")

        # Draw boxes
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    # FPS
    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    # Show on screen
    cv2.imshow("Hailo Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
pipeline.stop()
cv2.destroyAllWindows()
