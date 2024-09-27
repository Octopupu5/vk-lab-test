from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch import nn
import torch.optim as optim
from timm import create_model

import tqdm.notebook as tq

from PIL import Image
from io import BytesIO
from base64 import b64decode

import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_picture_model(model, train_loader, optimizer, criterion):
    running_loss = 0.0
    model.train()
    loop = tq.tqdm(enumerate(train_loader), total=len(train_loader), leave=True, colour="steelblue")
    for batch_idx, data in loop:
        images = data["img"].to(device)
        targets = data["target"].to(device, dtype=torch.float).squeeze()
                
        outputs = model(images).squeeze()
        loss = criterion(outputs, targets)

        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model, running_loss / len(train_loader)

def eval_picture_model(model, test_loader, criterion):
    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader, 0):
            images = data["img"].to(device)
            targets = data["target"].to(device, dtype=torch.float).squeeze()
                    
            outputs = model(images).squeeze()
            loss = criterion(outputs, targets)
            running_loss += loss.item()

    return running_loss / len(test_loader)

def test_model(model, epochs, train_loader, test_loader, loss, loss_name, path_to_model):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    best_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model, train_loss = train_picture_model(model, train_loader, optimizer, loss)
        test_loss = eval_picture_model(model, test_loader, loss)

        print(f"Train {loss_name} = {train_loss:.4f}, Test {loss_name} = {test_loss:.4f}")
        if test_loss < best_loss:
            torch.save(model.state_dict(), path_to_model)

class PictureDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cell = self.df.iloc[idx]
        
        img = Image.open(BytesIO(b64decode(cell["photo"])))
        img = np.asarray(img.convert("RGB"))
        img = self.transform(image=img)["image"]
        
        target = cell["open_photo"] / cell["view"] if cell["view"] > 0 else 0

        return {
            "img" : img,
            "target" : target
        }
    
class BCEWeighted(nn.BCELoss):
    def __init__(self, alpha):
        super().__init__(reduction='none')
        self.alpha = alpha

    def forward(self, input, target):
        loss = super().forward(input, target)
        weights = torch.exp(self.alpha * target)
        weighted_loss = (loss * weights).mean()
        return weighted_loss

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.image_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
        self.image_model.classifier.fc = nn.Linear(in_features=1280, out_features=1, bias=True)
        self.answer = nn.Sigmoid()

    def forward(self, image_input):
        return self.answer(self.image_model(image_input))
    
class ConvNeXt(nn.Module):
    def __init__(self):
        super(ConvNeXt, self).__init__()
        convnext = create_model("convnext_tiny", pretrained=True)
        self.image_model = convnext
        self.image_model.head.fc = nn.Linear(in_features=768, out_features=1, bias=True)
        self.answer = nn.Sigmoid()

    def forward(self, image_input):
        return self.answer(self.image_model(image_input))

max_size = 224
transform = A.Compose([
    A.Resize(max_size, max_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
        
    A.RandomBrightnessContrast(p=0.75),
    A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
    A.OneOf([
        A.GaussNoise(var_limit=[10, 50]),
        A.GaussianBlur(),
        A.MotionBlur(),
    ], p=0.4),
    A.CoarseDropout(max_holes=2, max_width=int(max_size * 0.2), max_height=int(max_size * 0.2), mask_fill_value=0, p=0.5),
    
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(transpose_mask=True),
])