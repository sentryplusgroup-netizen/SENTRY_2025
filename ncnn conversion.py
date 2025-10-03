from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model 

model = YOLO("deer_detect.pt")  # load a pretrained model 

# Export the model to NCNN format
model.export(format="onnx", imgsz=320)  #creates yolov8n_ncnn_model  