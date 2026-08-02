from ultralytics import YOLO

# Load a YOLO26n PyTorch model
model = YOLO("yolo26n.pt")

# Export the model to TensorRT with DLA enabled (only works with FP16 or INT8)
model.export(format="engine", device="dla:0", quantize=16)  # dla:0 or dla:1 corresponds to the DLA cores

# Load the exported TensorRT model
trt_model = YOLO("yolo26n.engine")

# Run inference
results = trt_model("https://ultralytics.com/images/bus.jpg")

