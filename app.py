import torch
from torchvision import models, transforms
from PIL import Image
import json
from fastapi import FastAPI, UploadFile
from io import BytesIO
from fastapi.middleware.cors import CORSMiddleware

# Load labels
with open("classes.json") as f:
    classes = json.load(f)

# Load model
model = models.resnet18()
num_classes = len(classes)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("anime_model.pth", map_location="cpu"))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow your frontend to call
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict(file: UploadFile):
    img = Image.open(BytesIO(await file.read())).convert("RGB")
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(x)
        _, pred = torch.max(outputs, 1)
    return {"prediction": classes[pred.item()]}
