import torch.nn as nn
import torch
from .blocks import DownConvBlock, UpConvBlock

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv_in = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1),
                                 bias=False)

        self.down1 = DownConvBlock(64, 128)
        self.down2 = DownConvBlock(128, 256)
        self.down3 = DownConvBlock(256, 512)
        self.down4 = DownConvBlock(512, 512)
        self.down5 = DownConvBlock(512, 512)
        self.down6 = DownConvBlock(512, 512)

        self.middle = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.up1 = UpConvBlock(1024, 512, dropout=True)
        self.up2 = UpConvBlock(1024, 512, dropout=True)
        self.up3 = UpConvBlock(1024, 512, dropout=True)
        self.up4 = UpConvBlock(1024, 256)
        self.up5 = UpConvBlock(512, 128)
        self.up6 = UpConvBlock(256, 64)

        self.outermost = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 3, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Tanh()
        )

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)

        x = self.middle(x6)

        x = self.up1(torch.cat((x, x6), dim=1))
        x = self.up2(torch.cat((x, x5), dim=1))
        x = self.up3(torch.cat((x, x4), dim=1))
        x = self.up4(torch.cat((x, x3), dim=1))
        x = self.up5(torch.cat((x, x2), dim=1))
        x = self.up6(torch.cat((x, x1), dim=1))
        return self.outermost(torch.cat((x, x0), dim=1))
