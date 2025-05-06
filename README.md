# YOLOv8 FastAPI Backend

This is a simple FastAPI server using YOLOv8 to detect objects in uploaded images.

## Setup Locally

```bash
pip install -r requirements.txt
uvicorn yolo_server:app --host 0.0.0.0 --port 8000
