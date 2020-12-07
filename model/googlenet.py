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

class GoogleNet(nn.Module):
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

        self.layers = nn.ModuleList([
            self.conv_layer1,
            self.mp_layer1,
            self.conv_layer2,
            self.mp_layer2,
            self.conv_layer3, 
            self.conv_layer4, 
            self.conv_layer5, 
            self.conv_layer6, 
            self.mp_layer3,
            self.conv_layer7, 
            self.conv_layer8, 
            self.conv_layer9, 
            self.conv_layer10,
            self.conv_layer11,
            self.conv_layer12,
            self.conv_layer13,
            self.conv_layer14,
            self.conv_layer15,
            self.conv_layer16,
            self.mp_layer4,
            self.conv_layer17,
            self.conv_layer18,
            self.conv_layer19,
            self.conv_layer20,
            self.conv_layer21,
            self.conv_layer22,
            self.conv_layer23,
            self.conv_layer24
        ])

    def debug(self, x):
        shape_x = list(x.shape)[1:]
        # print(f'shape {shape_x}')
        return x, shape_x

    def forward(self, x, test=False, test_set=[]):
        if test:
            shapes = []
            for layer in self.layers:
                x, shape_x = self.debug(layer(x))
                shapes.append(shape_x)
            for shape_out, shape_test in zip(shapes, test_set):
                print(shape_out, shape_test)
                assert shape_out == shape_test
                '''test failed'''
        else:
            for layer in self.layers:
                x = layer(x)
        return x


if __name__ == "__main__":
    model = GoogleNet()
    test_input = torch.zeros([1, 3, 448, 448])
    test_output_size=[
        # [3,448,448],
        [64,224,224],
        [64,112,112],
        [192,112,112],
        [192,56,56],
        [128,56,56],
        [256,56,56],
        [256,56,56],
        [512,56,56],
        [512,28,28],
        [256,28,28],
        [512,28,28],
        [256,28,28],
        [512,28,28],
        [256,28,28],
        [512,28,28],
        [256,28,28],
        [512,28,28],
        [512,28,28],
        [1024,28,28],
        [1024,14,14],
        [512,14,14],
        [1024,14,14],
        [512,14,14],
        [1024,14,14],
        [1024,7, 7]
    ]
    test_output = model(test_input, test = True, test_set = test_output_size)
