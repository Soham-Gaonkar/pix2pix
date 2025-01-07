import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train(netD, netG, optimizer_G, optimizer_D, scheduler_G, scheduler_D, train_dataloader, val_dataloader, NB_EPOCHS, device, lambda_L1=100, criterionL1=torch.nn.L1Loss()):
    writer = SummaryWriter(log_dir='./runs/facades_experiment')
    best_val_loss = float("inf")
    print('#'*50)
    print('Starting training ...')
    print('#'*50)  
    for epoch in tqdm(range(NB_EPOCHS)):
        netD.train()
        netG.train()
        train_loss_G, train_loss_D = 0.0, 0.0
        
        for batch in train_dataloader:
            real_A, real_B = batch
            real_A, real_B = real_A.to(device), real_B.to(device)

            fake_B = netG(real_A)

            # Discriminator Loss
            pred_fake = netD(torch.cat((real_A, fake_B), 1).detach())
            loss_D_fake = torch.nn.functional.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = netD(real_AB)
            loss_D_real = torch.nn.functional.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Generator Loss
            pred_fake = netD(torch.cat((real_A, fake_B), 1))
            loss_G_GAN = torch.nn.functional.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = lambda_L1 * criterionL1(fake_B, real_B)
            loss_G = loss_G_GAN + loss_G_L1
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            train_loss_D += loss_D.item()
            train_loss_G += loss_G.item()
        
        scheduler_D.step()
        scheduler_G.step()

        # Validation
        netD.eval()
        netG.eval()
        val_loss_G, val_loss_L1 = 0.0, 0.0

        with torch.no_grad():
            for batch in val_dataloader:
                real_A, real_B = batch
                real_A, real_B = real_A.to(device), real_B.to(device)

                fake_B = netG(real_A)

                loss_G_L1 = criterionL1(fake_B, real_B)
                val_loss_L1 += loss_G_L1.item()

        writer.add_scalar('Loss/Train Generator', train_loss_G / len(train_dataloader), epoch)
        writer.add_scalar('Loss/Train Discriminator', train_loss_D / len(train_dataloader), epoch)
        writer.add_scalar('Loss/Validation L1', val_loss_L1 / len(val_dataloader), epoch)

        print(f"Epoch {epoch + 1}/{NB_EPOCHS}")
        print(f"Train Loss G: {train_loss_G / len(train_dataloader):.4f}, Train Loss D: {train_loss_D / len(train_dataloader):.4f}")
        print(f"Val Loss L1: {val_loss_L1 / len(val_dataloader):.4f}")

        if val_loss_L1 < best_val_loss:
            best_val_loss = val_loss_L1
            torch.save(netG.state_dict(), "best_generator.pth")
            torch.save(netD.state_dict(), "best_discriminator.pth")
            print("Saved best model")
    
    writer.close()
