from flask import Flask, request, jsonify
import torch
from flask_cors import CORS
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

app = Flask(__name__)
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("trained_model.pth", map_location=DEVICE))
model.eval()
model.to(DEVICE)

classes = ["Clear Region", "Contaminated Region"]

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    img = Image.open(request.files["image"])
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)[0]
        idx = probs.argmax().item()

    return jsonify({
        "prediction": classes[idx],
        "confidence": round(float(probs[idx]) * 100, 2)
    })

if __name__ == "__main__":
    app.run(debug=True)
