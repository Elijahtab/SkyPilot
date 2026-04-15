import kagglehub

# Download latest version
path = kagglehub.dataset_download("ryankraus/traffic-camera-object-detection")

print("Path to dataset files:", path)
