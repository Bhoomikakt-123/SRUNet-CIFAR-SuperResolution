import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform_input = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_target = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_input)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_input)

train_loader = DataLoader(trainset, batch_size=8, shuffle=True)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

class SRUNet(nn.Module):
    def __init__(self):
        super(SRUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = SRUNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for data, _ in train_loader:
        input_images = data.to(device)
        targets = torch.stack([transform_target(transforms.ToPILImage()(img.cpu())) for img in input_images]).to(device)
        optimizer.zero_grad()
        output = model(input_images)
        output_resized = nn.functional.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)
        loss = criterion(output_resized, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), "models/srunet_model.pth")

def calculate_psnr(target, output):
    mse = np.mean((target - output) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))

def calculate_ssim(target, output):
    return ssim(target, output, channel_axis=-1)

model.eval()
psnr_vals, ssim_vals = [], []
with torch.no_grad():
    for data, _ in test_loader:
        input_images = data.to(device)
        targets = torch.stack([transform_target(transforms.ToPILImage()(img.cpu())) for img in input_images]).to(device)
        output = model(input_images)
        output_resized = nn.functional.interpolate(output, size=(128, 128), mode='bilinear', align_corners=False)

        targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
        outputs_np = output_resized.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5
        inputs_np = input_images.cpu().numpy().transpose(0, 2, 3, 1) * 0.5 + 0.5

        for i in range(len(outputs_np)):
            psnr_vals.append(calculate_psnr(targets_np[i], outputs_np[i]))
            ssim_vals.append(calculate_ssim(targets_np[i], outputs_np[i]))

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(np.clip(inputs_np[0], 0, 1)); ax[0].set_title("Low-Res Input"); ax[0].axis("off")
        ax[1].imshow(np.clip(targets_np[0], 0, 1)); ax[1].set_title("High-Res Target"); ax[1].axis("off")
        ax[2].imshow(np.clip(outputs_np[0], 0, 1)); ax[2].set_title("Model Output"); ax[2].axis("off")
        plt.tight_layout(); plt.savefig("outputs/sample_output.png"); plt.show()
        break

print(f"📊 Average PSNR: {np.mean(psnr_vals):.2f} dB")
print(f"📊 Average SSIM: {np.mean(ssim_vals):.4f}")
