import torch.nn as nn

class PatchGAN(nn.Module):

    def __init__(self):
        super(PatchGAN, self).__init__()

        self.network = nn.Sequential(
                            nn.Conv2d(6, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(64, 128, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(256, 512, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1), bias=False),
                            nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                            nn.LeakyReLU(negative_slope=0.2, inplace=True),
                            nn.Conv2d(512, 1, kernel_size=(4, 4), stride=(1, 1), padding=(1, 1)))

    def forward(self, x):
        return self.network(x)