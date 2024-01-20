from ultralytics import YOLO

# Load a model
model = YOLO("yolov5s.yaml")  # build a new model from scratch
model = YOLO("yolov5s.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="config.yaml", epochs=3)  # train the model