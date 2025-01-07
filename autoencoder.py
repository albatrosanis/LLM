import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Global Parameters
batchSize = 64
lr = 1e-4
nepoch = 10
root = "./datasets"
noise_scale = 0.3
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Data Transforms
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load MNIST Dataset
train_set = Datasets.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(train_set, batch_size=batchSize, shuffle=True, num_workers=0)

test_set = Datasets.MNIST(root=root, train=False, transform=transform, download=True)
test_loader = DataLoader(test_set, batch_size=batchSize, shuffle=False, num_workers=0)

# Encoder Network
class Encoder(nn.Module):
    def __init__(self, channels, ch=32, z=32):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=ch, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(in_channels=ch, out_channels=ch * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch * 2)
        self.conv3 = nn.Conv2d(in_channels=ch * 2, out_channels=ch * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch * 4)
        self.conv_out = nn.Conv2d(in_channels=ch * 4, out_channels=z, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.conv_out(x)
        return x

# Decoder Network
class Decoder(nn.Module):
    def __init__(self, channels, ch=32, z=32):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels=z, out_channels=ch * 4, kernel_size=4, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(ch * 4)
        self.conv2 = nn.ConvTranspose2d(in_channels=ch * 4, out_channels=ch * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ch * 2)
        self.conv3 = nn.ConvTranspose2d(in_channels=ch * 2, out_channels=ch, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(ch)
        self.conv4 = nn.ConvTranspose2d(in_channels=ch, out_channels=channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.tanh(self.conv4(x))
        return x

# Autoencoder
class AE(nn.Module):
    def __init__(self, channel_in, ch=16, z=32):
        super(AE, self).__init__()
        self.encoder = Encoder(channels=channel_in, ch=ch, z=z)
        self.decoder = Decoder(channels=channel_in, ch=ch, z=z)

    def forward(self, x):
        encoding = self.encoder(x)
        x = self.decoder(encoding)
        return x, encoding

# Create Network
latent_size = 128
ae_net = AE(channel_in=1, ch=32, z=latent_size).to(device)
optimizer = optim.Adam(ae_net.parameters(), lr=lr)
loss_func = nn.MSELoss()

# Training Loop
loss_log = []
for epoch in range(nepoch):
    ae_net.train()
    train_loss = 0
    for data in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{nepoch}"):
        image = data[0].to(device)
        random_sample = (torch.bernoulli((1 - noise_scale) * torch.ones_like(image)) * 2) - 1
        noisy_img = random_sample * image

        recon_data, _ = ae_net(noisy_img)
        loss = loss_func(recon_data, image)
        loss_log.append(loss.item())
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{nepoch}] Loss: {train_loss/len(train_loader):.4f}")

# Visualization
def visualize_results(test_loader, model):
    model.eval()
    dataiter = iter(test_loader)
    test_images, _ = next(dataiter)

    # Noisy Images
    random_sample = (torch.bernoulli((1 - noise_scale) * torch.ones_like(test_images)) * 2) - 1
    noisy_test_img = random_sample * test_images

    # Reconstruction
    recon_data, _ = model(noisy_test_img.to(device))

    # Plotting
    fig, axes = plt.subplots(3, 8, figsize=(20, 10))
    for i in range(8):
        axes[0, i].imshow(test_images[i, 0].numpy(), cmap='gray')
        axes[1, i].imshow(noisy_test_img[i, 0].numpy(), cmap='gray')
        axes[2, i].imshow(recon_data[i, 0].detach().cpu().numpy(), cmap='gray')

    plt.show()

visualize_results(test_loader, ae_net)
