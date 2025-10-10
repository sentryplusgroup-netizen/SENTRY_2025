import cv2
from ultralytics import YOLO

# Open the USB camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)

# Load YOLOv8 model
model = YOLO("yolo11n_ncnn_model")

# Define tracker type (optional, default is 'bytetrack.yaml')
tracker = "bytetrack.yaml"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Run YOLO model with tracking
    results = model.track(frame, persist=True, tracker=tracker)

    # Annotate frame with bounding boxes and IDs
    annotated_frame = results[0].plot()

    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time
    text = f'FPS: {fps:.1f}'

    # Draw FPS on frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10
    text_y = text_size[1] + 10
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display annotated frame
    cv2.imshow("USB Camera Tracking", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows() 