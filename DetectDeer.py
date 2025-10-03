import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

# Configuration
MODEL_PATH = "deer_detect_ncnn_model"
TARGET_LABELS = {"deer"}  # add other labels if needed
IMG_SIZE = 320

# Set up the camera with Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (IMG_SIZE, IMG_SIZE)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load custom YOLO model (explicit task avoids the warning)
model = YOLO(MODEL_PATH, task="detect")

try:
    while True:
        # Capture a frame from the camera
        frame = picam2.capture_array()

        # Run YOLO model on the captured frame
        try:
            results = model.predict(frame, imgsz=IMG_SIZE)
        except Exception as e:
            print(f"Model predict error: {e}")
            continue

        # Determine detected class names (robust to different result types)
        detected_names = []
        try:
            cls_obj = results[0].boxes.cls  # tensor or list-like
            cls_list = cls_obj.tolist() if hasattr(cls_obj, "tolist") else list(cls_obj)
            names_map = results[0].names if hasattr(results[0], "names") else {}
            for c in cls_list:
                idx = int(c)
                if isinstance(names_map, dict):
                    detected_names.append(names_map.get(idx, str(idx)))
                else:
                    # guard against out-of-range / unexpected names_map
                    try:
                        detected_names.append(names_map[idx])
                    except Exception:
                        detected_names.append(str(idx))
        except Exception:
            # fallback: treat any boxes as unknown detections
            try:
                if len(results[0].boxes):
                    detected_names = ["unknown"] * len(results[0].boxes)
            except Exception:
                detected_names = []

        # Only proceed/display if a target label (e.g., "deer") is detected
        if not any(name in TARGET_LABELS for name in detected_names):
            # optionally close window if it exists
            try:
                cv2.destroyWindow("Camera")
            except Exception:
                pass
            continue

        # Annotate frame from results
        annotated_frame = results[0].plot()

        # Ensure we display with OpenCV (BGR). Picamera2 / ultralytics often use RGB.
        try:
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        except Exception:
            display_frame = annotated_frame  # fallback

        # Get inference time (guard against zero/missing)
        inference_time = 0
        try:
            inference_time = results[0].speed.get("inference", 0) if hasattr(results[0], "speed") else 0
        except Exception:
            inference_time = 0
        fps = (1000.0 / inference_time) if inference_time and inference_time > 0 else 0.0
        text = f"FPS: {fps:.1f}"

        # Draw FPS text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1, 2)[0]
        text_x = max(10, display_frame.shape[1] - text_size[0] - 10)
        text_y = text_size[1] + 10
        cv2.putText(display_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the resulting frame (only when target detected)
        cv2.imshow("Camera", display_frame)
        if cv2.waitKey(1) == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    try:
        picam2.stop()
    except Exception:
        pass
    try:
        picam2.close()
    except Exception:
        pass
    cv2.destroyAllWindows()