from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model

model = YOLO("Sentry_finModel_1.pt")  # load a pretrained model 

# Export the model to NCNN format
model.export(format="ncnn", imgsz=480, task="segment")  #creates yolov8n_ncnn_model  