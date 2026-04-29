License Plate Detection & Recognition Pipeline

This project implements an end-to-end computer vision pipeline for detecting vehicles, localizing license plates, and recognizing plate text from video input. It uses YOLOv8 for object detection, SORT for multi-object tracking, and OCR for extracting license plate numbers. The system processes each video frame, associates detected plates with tracked vehicles, and stores structured results while also providing real-time visual output with bounding boxes and annotations.

The pipeline is designed to be efficient and robust, handling edge cases such as empty detections, invalid crops, and OCR failures. It also includes preprocessing techniques to improve text recognition accuracy. The final output is saved in CSV format for further analysis or integration into downstream systems.

Key Features
Real-time vehicle detection using YOLOv8
License plate detection and association with tracked vehicles
Multi-object tracking using SORT
OCR-based license plate recognition with preprocessing improvements
Robust handling of edge cases (empty detections, invalid crops)
Live visualization with bounding boxes and labels
Structured output saved as CSV

Tech Stack
Python
OpenCV
Ultralytics YOLOv8
NumPy
SORT Tracker
EasyOCR
How to Run

Install dependencies:

pip install ultralytics opencv-python numpy easyocr
Clone SORT repository and place sort.py in the project folder
Add required models:
yolov8n.pt
license_plate_detector.pt
Run the script:
python main.py
Output
Annotated video display showing detected vehicles and license plates
Extracted license plate data saved in test.csv
