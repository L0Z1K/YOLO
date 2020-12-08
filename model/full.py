import torch
from torch import nn

from pretrain import Pretrain, ConvBlock

class YOLO(nn.Module):
    def __init__(self, n_box, n_class):
        self.n_box = n_box
        self.n_class = n_class
        super().__init__()
        self.layers = nn.ModuleList([
            Pretrain(),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 2, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
            ConvBlock(1024, 1024, 3, 1, 1),
        ])
        self.linear1 = nn.Linear(1024*7*7, 4096)
        self.linear2 = nn.Linear(4096, (5 * n_box + n_class)*7*7)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = torch.reshape(x, (-1, 7, 7, 30))
        p_class = x[:, -self.n_class:]
        print(p_class.shape)
        for i in n_box:
            
        return x    

if __name__ == "__main__":
    model = YOLO(n_box = 2, n_class = 20)

    test_input = torch.zeros([2, 3, 448, 448])
    test_output = model(test_input)
    print(test_output.shape)