from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Enable CORS for Flutter or web apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your Flutter app domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 nano model for lightweight detection
model = YOLO("yolov8n.pt")

@app.post("/detect/")
async def detect(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    results = model(image, conf=0.4)[0]
    detections = []
    for box in results.boxes:
        label = results.names[int(box.cls)]
        conf = float(box.conf)
        detections.append({
            "label": label,
            "confidence": conf
        })

    return {"detections": detections}
