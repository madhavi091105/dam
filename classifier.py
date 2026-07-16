import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Agar tere actual class names alag hain, yahan update kar de
DEFAULT_CLASS_NAMES = ["Easy", "Medium", "Hard"]


class DamageClassifier(nn.Module):
    """Matches checkpoint keys: backbone.features.* + backbone.classifier.{0..5}"""
    def __init__(self, num_classes, hidden_dim=512, backbone_name="efficientnet_b0"):
        super().__init__()
        backbone_fn = getattr(models, backbone_name)
        self.backbone = backbone_fn(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def load_classifier(ckpt_path, device="cpu"):
    state_dict = torch.load(ckpt_path, map_location=device)
    num_classes = state_dict["backbone.classifier.5.weight"].shape[0]
    hidden_dim = state_dict["backbone.classifier.1.weight"].shape[0]

    model = DamageClassifier(num_classes=num_classes, hidden_dim=hidden_dim)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, num_classes


CLS_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict(model, image: Image.Image, device="cpu", class_names=None):
    x = CLS_TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    pred_idx = int(probs.argmax())

    names = class_names or [str(i) for i in range(len(probs))]
    if class_names is None and len(probs) == len(DEFAULT_CLASS_NAMES):
        names = DEFAULT_CLASS_NAMES

    label = names[pred_idx]
    return {
        "label": label,
        "confidence": float(probs[pred_idx]),
        "probs": {names[i]: float(probs[i]) for i in range(len(probs))},
    }