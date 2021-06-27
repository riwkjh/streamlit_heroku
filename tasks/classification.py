from torchvision import models, transforms
import torch
import torch.nn as nn
import pathlib
from PIL import Image

upper_path = pathlib.Path().resolve()

model_path = upper_path / "models" / "classification_model.pth"
cls_list = ['Black Sea Sprat',
            'Gilt-Head Bream',
            'Hourse Mackerel',
            'Red Mullet',
            'Red Sea Bream',
            'Sea Bass',
            'Shrimp',
            'Striped Red Mullet',
            'Trout']
rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

def predict(image_path):
    model = models.mobilenet_v3_small(pretrained=False)
    model.classifier[3] = nn.Linear(1024, len(cls_list))
    model.load_state_dict(torch.load(model_path))

    # https://pytorch.org/docs/stable/torchvision/models.html
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=rgb_mean,
            std=rgb_std
        )])

    img = Image.open(image_path)
    batch_t = torch.unsqueeze(transform(img), 0)

    model.eval()
    out = model(batch_t)

    prob = torch.nn.functional.softmax(out, dim=1)[0] * 100
    _, indices = torch.sort(out, descending=True)
    return [(cls_list[idx], prob[idx].item()) for idx in indices[0][:5]]
