from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model

model = YOLO("SentryYOLOv8_1.pt")  # load a pretrained model 

# Export the model to NCNN format
model.export(format="ncnn", imgsz=640, task="segment")  #creates yolov8n_ncnn_model  