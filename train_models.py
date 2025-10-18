from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="Deer Segmentation.v1-sentryv1_seg.yolov8.zip", epochs=20,patience=70, imgsz=480, name="yolov8n_custom_model")

