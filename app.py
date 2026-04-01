import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import gradio as gr

# -----------------------------
# MODEL DEFINITION
# -----------------------------
class Attention(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.query = nn.Linear(in_features, in_features)
        self.key = nn.Linear(in_features, in_features)
        self.value = nn.Linear(in_features, in_features)

    def forward(self, x):
        # x: (batch, features)
        Q = self.query(x).unsqueeze(1)  # (batch, 1, features)
        K = self.key(x).unsqueeze(1)
        V = self.value(x).unsqueeze(1)

        # Scaled dot-product attention
        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (x.size(1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_output = torch.bmm(attn_weights, V).squeeze(1)
        return attn_output

class ResNetAttention(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.attn = Attention(512)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.attn(x)
        x = self.classifier(x)
        return x

# -----------------------------
# LOAD MODEL
# -----------------------------
model = ResNetAttention(num_classes=3)
# Make sure this path points to the correct full weight file!
model.load_state_dict(torch.load("resnet_attention_weights (1).pth", map_location="cpu"))
model.eval()

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # match training
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # match training
])

# -----------------------------
# CLASSES
# -----------------------------
classes = [
    "History of MI",
    "Abnormal Heartbeat",
    "Normal ECG"
]

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict(image: Image.Image):
    image = transform(image).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

        # Optional debug prints
        print("Raw outputs:", outputs)
        print("Probabilities:", probs)

    return f"{classes[predicted.item()]} ({confidence.item()*100:.2f}%)"

# -----------------------------
# GRADIO INTERFACE
# -----------------------------
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="ECG Classification",
    description="Upload an ECG image and predict the class"
)

# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    interface.launch()