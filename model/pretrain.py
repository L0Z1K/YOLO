from torch import nn
import torch
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    
    def forward(self, x):
        return self.model(x)

class Pretrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([               # output = (input-kernel+2*pad)/stride + 1
            ConvBlock(3, 64, 7, 2, 3),              # 3, 448, 448 -> 64, 224, 224
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64, 224, 224 -> 64, 112, 112
            ConvBlock(64, 192, 3, 1, 1),            # 64, 112, 112 -> 192, 112, 112
            nn.MaxPool2d(2, 2),                     # 192, 112, 112 -> 192, 56, 56
            ConvBlock(192, 128, 1, 1, 0),           # 192, 56, 56 -> 128, 56, 56
            ConvBlock(128, 256, 3, 1, 1),           # 128, 56, 56 -> 256, 56, 56
            ConvBlock(256, 256, 1, 1, 0),           # 256, 56, 56 -> 256, 56, 56
            ConvBlock(256, 512, 3, 1, 1),           # 256, 56, 56 -> 512, 56, 56
            nn.MaxPool2d(2, 2),                     # 512, 56, 56 -> 512, 28, 28
            ConvBlock(512, 256, 1, 1, 0),           # 512, 28, 28 -> 256, 28, 28
            ConvBlock(256, 512, 3, 1, 1),           # 256, 28, 28 -> 512, 28, 28
            ConvBlock(512, 256, 1, 1, 0),           # 512, 28, 28 -> 256, 28, 28
            ConvBlock(256, 512, 3, 1, 1),           # 256, 28, 28 -> 512, 28, 28
            ConvBlock(512, 256, 1, 1, 0),           # 512, 28, 28 -> 256, 28, 28
            ConvBlock(256, 512, 3, 1, 1),           # 256, 28, 28 -> 512, 28, 28
            ConvBlock(512, 256, 1, 1, 0),           # 512, 28, 28 -> 256, 28, 28
            ConvBlock(256, 512, 3, 1, 1),           # 256, 28, 28 -> 512, 28, 28
            ConvBlock(512, 512, 1, 1, 0),           # 512, 28, 28 -> 512, 28, 28
            ConvBlock(512, 1024, 3, 1, 1),          # 512, 28, 28 -> 1024, 28, 28
            nn.MaxPool2d(2,2),                      # 1024, 28, 28 -> 1024, 14, 14
            ConvBlock(1024, 512, 1, 1, 0),          # 1024, 14, 14 -> 512, 14, 14
            ConvBlock(512, 1024, 3, 1, 1),          # 512, 14, 14 -> 1024, 14, 14
            ConvBlock(1024, 512, 1, 1, 0),          # 1024, 14, 14 -> 512, 14, 14
            ConvBlock(512, 1024, 3, 1, 1),          # 512, 14, 14 -> 1024, 14, 14
            # For pretraining we use the first 20 convolutional layers
        ])

    def debug(self, layer, x):
        out = layer(x)
        print(f'{list(x.shape)}-{layer}->{list(out.shape)}')
        return out

    def forward(self, x, test=False):
        if test:
            for layer in self.layers:
                x = self.debug(layer, x)
        else:
            for layer in self.layers:
                x = layer(x)
        return x


if __name__ == "__main__":
    model = Pretrain()
    test_input = torch.zeros([1, 3, 448, 448])
    test_output = model(test_input)
    print(test_output.shape)