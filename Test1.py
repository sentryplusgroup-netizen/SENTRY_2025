import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2

# ------------------------------
# CONFIGURATION
# ------------------------------
MODEL_PATH = "deer_detect.onnx"
CONF_THRESHOLD = 0.5   # Confidence threshold
LINE_X = 320           # Initial line position (middle of 640px width)

# ------------------------------
# LOAD ONNX MODEL
# ------------------------------
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [batch, channels, height, width]
output_name = session.get_outputs()[0].name

# ------------------------------
# SETUP CAMERA
# ------------------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# ------------------------------
# GLOBALS FOR SWEEP LINE
# ------------------------------
window_name = "Deer Detector Sweep Line"
cv2.namedWindow(window_name)
dragging = False
seen_ids = set()
total_count = 0

# ------------------------------
# MOUSE CALLBACK FOR DRAGGABLE LINE
# ------------------------------
def drag_line(event, x, _, flags, param):
    global LINE_X, dragging
    if event == cv2.EVENT_LBUTTONDOWN or (flags & cv2.EVENT_FLAG_LBUTTON):
        LINE_X = max(0, min(x, 640))
        dragging = True

cv2.setMouseCallback(window_name, drag_line)

# ------------------------------
# HELPER: PREPROCESS IMAGE
# ------------------------------
def preprocess(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (input_shape[3], input_shape[2]))
    img_input = img_resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return img_input

# ------------------------------
# MAIN LOOP
# ------------------------------
while True:
    frame = picam2.capture_array()
    img_input = preprocess(frame)

    # Run inference
    outputs = session.run([output_name], {input_name: img_input})
    preds = outputs[0][0]  # remove batch dimension

    current_frame_ids = set()

    for det in preds:
        # Check if output is [x1, y1, x2, y2, conf, class]
        if len(det) != 6:
            continue
        x1, y1, x2, y2, conf, cls = det
        cls = int(cls)
        if conf < CONF_THRESHOLD:
            continue
        if cls != 0:  # assume class 0 = deer
            continue

        # Assign a pseudo ID based on box coordinates (for counting)
        t_id = int(x1 + y1 + x2 + y2)  # simple hash
        current_frame_ids.add(t_id)

        # Count if deer crosses line
        if x1 > LINE_X and t_id not in seen_ids:
            seen_ids.add(t_id)
            total_count += 1

        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"Deer {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw sweep line and total count
    cv2.line(frame, (LINE_X, 0), (LINE_X, 480), (0, 0, 255), 2)
    cv2.putText(frame, f"TOTAL: {total_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
