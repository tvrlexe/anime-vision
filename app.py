import torch
from torchvision import models, transforms
from PIL import Image
import json
from fastapi import FastAPI, UploadFile
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import os

# ---- Load class labels ----
try:
    with open("classes.json", "r") as f:
        classes = json.load(f)
except Exception as e:
    classes = []
    print("Warning: Could not load classes.json:", e)

# ---- Load trained model ----
model = models.resnet18(weights=None)
num_classes = len(classes) if classes else 1000  # fallback
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

try:
    model.load_state_dict(torch.load("anime_model.pth", map_location="cpu"))
    print("Model loaded successfully.")
except Exception as e:
    print("Warning: Could not load model weights:", e)

model.eval()

# ---- Image transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---- Initialize FastAPI app ----
app = FastAPI(
    title="Anime Classifier",
    description="Upload an anime image to get the predicted title/class",
    version="1.0",
)

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Serve Frontend (index.html) ----
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
def serve_index():
    return FileResponse("index.html")

# ---- Prediction endpoint ----
@app.post("/predict")
async def predict(file: UploadFile):
    try:
        img = Image.open(BytesIO(await file.read())).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(x)
            _, pred = torch.max(outputs, 1)
        predicted_class = classes[pred.item()] if classes else f"Class {pred.item()}"
        return {"prediction": predicted_class}
    except Exception as e:
        return {"error": str(e)}

# ---- Optional local run ----
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
