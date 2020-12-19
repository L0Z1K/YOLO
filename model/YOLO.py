import torch
import torch.nn as nn

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


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layer1 = ConvBlock(3, 64, 7, 2, 3)            # 3*448*448 -> 64*224*224
        self.mp_layer1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 64*224*224 -> 64*112*112
        self.conv_layer2 = ConvBlock(64, 192, 3, 1, 1)          # 64*112*112 -> 192*112*112
        self.mp_layer2 = nn.MaxPool2d(2, 2)                     # 192*112*112 -> 192*56*56
        self.conv_layer3 = ConvBlock(192, 128, 1, 1, 0)         # 192*56*56 -> 128*56*56
        self.conv_layer4 = ConvBlock(128, 256, 3, 1, 1)         # 128*56*56 -> 256*56*56
        self.conv_layer5 = ConvBlock(256, 256, 1, 1, 0)         # 256*56*56 -> 256*56*56
        self.conv_layer6 = ConvBlock(256, 512, 3, 1, 1)         # 256*56*56 -> 512*56*56
        self.mp_layer3 = nn.MaxPool2d(2, 2)                     # 512*56*56 -> 512*28*28
        self.conv_layer7 = ConvBlock(512, 256, 1, 1, 0)         # 512*28*28 -> 256*28*28
        self.conv_layer8 = ConvBlock(256, 512, 3, 1, 1)         # 256*28*28 -> 512*28*28
        self.conv_layer9 = ConvBlock(512, 256, 1, 1, 0)         # 512*28*28 -> 256*28*28
        self.conv_layer10 = ConvBlock(256, 512, 3, 1, 1)        # 256*28*28 -> 512*28*28        
        self.conv_layer11 = ConvBlock(512, 256, 1, 1, 0)        # 512*28*28 -> 256*28*28
        self.conv_layer12 = ConvBlock(256, 512, 3, 1, 1)        # 256*28*28 -> 512*28*28        
        self.conv_layer13 = ConvBlock(512, 256, 1, 1, 0)        # 512*28*28 -> 256*28*28
        self.conv_layer14 = ConvBlock(256, 512, 3, 1, 1)        # 256*28*28 -> 512*28*28
        self.conv_layer15 = ConvBlock(512, 512, 1, 1, 0)        # 512*28*28 -> 512*28*28
        self.conv_layer16 = ConvBlock(512, 1024, 3, 1, 1)       # 512*28*28 -> 1024*28*28
        self.mp_layer4 = nn.MaxPool2d(2,2)                      # 1024*28*28 -> 1024*14*14
        self.conv_layer17 = ConvBlock(1024, 512, 1, 1, 0)       # 1024*14*14 -> 512*14*14
        self.conv_layer18 = ConvBlock(512, 1024, 3, 1, 1)       # 512*14*14 -> 1024*14*14
        self.conv_layer19 = ConvBlock(1024, 512, 1, 1, 0)       # 1024*14*14 -> 512*14*14
        self.conv_layer20 = ConvBlock(512, 1024, 3, 1, 1)       # 512*14*14 -> 1024*14*14
        self.conv_layer21 = ConvBlock(1024, 1024, 3, 1, 1)      
        self.conv_layer22 = ConvBlock(1024, 1024, 3, 2, 1)      # 1024*14*14 -> 1024*7*7
        self.conv_layer23 = ConvBlock(1024, 1024, 3, 1, 1)
        self.conv_layer24 = ConvBlock(1024, 1024, 3, 1, 1)

        self.linear1 = nn.Linear(1024*7*7, 4096)
        self.linear2 = nn.Linear(4096, 30*7*7)
    
    def forward(self, x):
        pass

if __name__ == "__main__":
    pass
        
