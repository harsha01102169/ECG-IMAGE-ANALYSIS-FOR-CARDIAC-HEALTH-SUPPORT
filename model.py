import torch
import torch.nn as nn
import torchvision.models as models

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