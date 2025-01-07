import torch.nn as nn

class DownConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        return self.block(x)

class UpConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False):
        super(UpConvBlock, self).__init__()
        layers = [
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        ]
        if dropout:
            layers.append(nn.Dropout(p=0.5, inplace=False))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)