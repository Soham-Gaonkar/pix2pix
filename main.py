import torch
from models.generator import UNet
from models.discriminator import PatchGAN
from datasets.dataset import Dataset
from training.train import train
from utils.weights import init_weights
from utils.schedulers import create_schedulers
from torch.utils.data import DataLoader

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    netG = UNet().to(device)
    netD = PatchGAN().to(device)

    netG.apply(init_weights)
    netD.apply(init_weights)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

    scheduler_G, scheduler_D = create_schedulers(optimizer_G, optimizer_D)

    train_dataloader = DataLoader(Dataset("data/facades/train"), batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(Dataset("data/facades/val"), batch_size=1, shuffle=False, num_workers=0)

    train(
        netD, netG, 
        optimizer_G, optimizer_D, 
        scheduler_G, scheduler_D, 
        train_dataloader, val_dataloader, 
        NB_EPOCHS=200, device=device
    )