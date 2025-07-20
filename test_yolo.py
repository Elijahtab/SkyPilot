import torch
from ultralytics import YOLO
from PIL import Image

Image.open("test.jpg").show()

# Check and print CUDA availability and version
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    print("PyTorch CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
else:
    print("Running on CPU only")

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # "n" = nano model

# Run inference on a sample image
results = model("C:/Users/elija/OneDrive/Documents/GitHub/SkyPilot/test.jpg")

# Display results
results[0].show()