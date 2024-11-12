import torch
import torch.nn as nn
import torch.nn.functional as F


class CXAS_ImageDecoder_Head(nn.Module):
    def __init__(self, in_channels, ngf, num_classes, norm='batch', constant = False):
        super().__init__()

        if constant:
            self.up1 = UpInit(in_channels,in_channels,in_channels, True, True)
            self.up2 = Up(2*in_channels,in_channels, True)
            self.up3 = Up(2*in_channels,in_channels, True)
            self.up4 = Up(2*in_channels,in_channels, True)

            self.out = OutConv(in_channels, num_classes)

        else:
            self.up1 = UpInit(in_channels //2, in_channels, in_channels //4, True)
            self.up2 = Up(in_channels // 4 * 2,  in_channels //8, True)
            self.up3 = Up(in_channels // 8 * 2,  in_channels // 32, True)
            self.up4 = Up(in_channels // 32 * 2,  ngf, True)

            self.out = OutConv(ngf, num_classes)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)



class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = F.interpolate
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.up(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UpInit(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, temp_channels, in_channels, out_channels, bilinear=True, hax=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = F.interpolate
        self.conv1 = Conv(in_channels, temp_channels)
        if hax:
            self.conv2 = DoubleConv(in_channels+temp_channels, out_channels, in_channels // 2)
        else:
            self.conv2 = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.up(x1, x2.shape[2:], mode='bilinear')
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
