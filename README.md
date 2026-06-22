# Dispatch IQ - Gunny Bag Label Monitoring System

Dispatch IQ is an AI-based dispatch monitoring prototype developed for detecting gunny bags, checking label/text presence using OCR, and raising an alert if no readable label/text is found.

## Features

- YOLOv8-based gunny bag detection
- EasyOCR-based label/text reading
- Alert logic for missing label/text
- Streamlit dashboard for image, video, and camera input
- FastAPI backend for image detection API
- ONNX model export
- Jetson CPU inference testing
- Dockerized backend and dashboard

## Project Structure

```text
dispatch-iq/
├── backend/
│   ├── api.py
│   └── detector_ocr.py
├── dashboard/
│   └── app.py
├── dataset/
├── models/
│   ├── gunny_bag_detector.pt
│   └── gunny_bag_detector.onnx
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
└── README.md
