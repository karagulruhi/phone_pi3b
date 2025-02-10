from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

# Export to NCNN format
model.export(format="ncnn")  # creates '/yolo11n_ncnn_model'