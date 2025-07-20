Drone Vehicle Detection with YOLOv8

This project uses a PyTorch-based YOLOv8 model to detect and classify different types of vehicles (cars, trucks, etc.) from drone footage.

Project Features
- Real-time object detection using YOLOv8
- Based on PyTorch

Setup Instructions

1.Clone the Repository

git clone https://github.com/YOUR-USERNAME/your-repo-name.git
cd your-repo-name

2.Create a Virtual Environment

Windows:
python -m venv myenv
.\myenv\Scripts\activate

macOS/Linux:
python3 -m venv myenv
source myenv/bin/activate

3.Install Dependencies

pip install -r requirements.txt

If requirements.txt is missing, install manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python

Running YOLOv8 Inference

1.Test Detection on Sample Image

python test_yolo.py

This runs YOLOv8 on a sample image and shows the detection result.

Directory Structure

your-repo/
├── test_yolo.py          (Quick test script for YOLOv8)
├── requirements.txt      (Python package list)
├── .gitignore            (Ignores virtual environment and cache)
└── README.md             (This file)

Notes

- Do not push the myenv/ folder to GitHub — it is already listed in .gitignore
- If you add new Python packages, update requirements.txt using:
  pip freeze > requirements.txt
- Always activate the virtual environment before running code

TODO / Custom Training

- Collect labeled data for specific car types (SUV, Sedan, Truck, etc.)
- Fine-tune YOLOv8 on a custom dataset using the Ultralytics training API