from ultralytics import YOLO

model = YOLO("yolo11s.pt")
model.export(format="openvino", half=True)  # Xuất ra thư mục yolo11s_openvino_model
