import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from models.generator import UNet
from models.discriminator import PatchGAN
from datasets.dataset import Dataset

# Define device (CUDA if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the trained models
def load_model(generator_path, discriminator_path):
    netG = UNet().to(device)  # Replace with your generator model
    netD = PatchGAN().to(device)  # Replace with your discriminator model

    netG.load_state_dict(torch.load(generator_path))
    netD.load_state_dict(torch.load(discriminator_path))

    return netG, netD

# Set up dataset and dataloader
def get_dataloader(batch_size=1, test_folder='datasets/facades/test'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    test_dataloader = DataLoader(Dataset("data/facades/test"), batch_size=1, shuffle=True, num_workers=0)


    return test_dataloader

# Visualize a batch of images
def visualize_images(real_A, real_B, fake_B):
    real_A = real_A.squeeze().cpu().detach().numpy()
    real_B = real_B.squeeze().cpu().detach().numpy()
    fake_B = fake_B.squeeze().cpu().detach().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    ax[0].imshow(real_A.transpose(1, 2, 0))  # Assuming the shape is (C, H, W)
    ax[0].set_title("Real A")
    ax[1].imshow(real_B.transpose(1, 2, 0))
    ax[1].set_title("Real B")
    ax[2].imshow(fake_B.transpose(1, 2, 0))
    ax[2].set_title("Generated Fake B")
    plt.show()

# Evaluate PSNR (Peak Signal-to-Noise Ratio)
import math

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

# Main testing function
def test_model(generator_path, discriminator_path, batch_size=1):
    
    print('#'*50)
    print('Starting training ...')
    print('#'*50)  
    
    # Load the models
    netG, netD = load_model(generator_path, discriminator_path)
    
    # Set model to evaluation mode
    netG.eval()
    netD.eval()
    
    # Get test dataloader
    test_dataloader = get_dataloader(batch_size)
    
    with torch.no_grad():
        for i, (real_A, real_B) in enumerate(test_dataloader):
            # Send data to the device (GPU/CPU)
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            
            # Generate fake_B from real_A
            fake_B = netG(real_A)
            
            # Visualize results
            visualize_images(real_A, real_B, fake_B)
            
            # Calculate PSNR
            psnr_value = psnr(fake_B, real_B)
            print(f"PSNR for this batch: {psnr_value:.4f}")

if __name__ == "__main__":
    # Paths to the saved models
    generator_path = 'best_discriminator.pth'
    discriminator_path = 'best_generator.pth'

    # Test the model
    test_model(generator_path, discriminator_path, batch_size=1)
