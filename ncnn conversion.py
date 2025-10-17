from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n.pt")  # load a pretrained model

model = YOLO("yolo11n_custom.pt")  # load a pretrained model 

# Export the model to NCNN format
model.export(format="ncnn", imgsz=640)  #creates yolov8n_ncnn_model  