# Human-Detection-and-Automated-Pedestrain-Counting

This project uses **YOLOv8** (You Only Look Once version 8) for real-time human detection and pedestrian counting from **images**, **videos**, and **live webcam feeds** using **Python** and **OpenCV**.

---

## 📌 Project Overview

The goal of this project is to detect humans (pedestrians) in a given media source (image/video/live stream) and display the total count in real time. It can be used for crowd monitoring, surveillance, traffic analytics, and smart city applications.

Two modes of operation are provided:
- `pd.py`: Detects pedestrians from an image or video file.
- `realtime.py`: Detects pedestrians using a live webcam feed.

---

## 🧠 Model Used

- **Model:** [YOLOv8n](https://github.com/ultralytics/ultralytics) (Nano version for fast performance)
- **Framework:** [Ultralytics YOLO](https://docs.ultralytics.com/)
- **Library:** OpenCV for image processing and rendering

---

## 🗂️ Project Structure

```plaintext
├── pd.py              # Detection from image or video file
├── realtime.py        # Live pedestrian detection via webcam
├── yolov8n.pt         # YOLOv8n pre-trained weights (downloaded automatically)
└── README.md          # Project documentation
