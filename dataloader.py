import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import datasets, transforms

class Dataset(data.Dataset):
    def __init__(self, name):
        if name == "ImageNet":
            transform = transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=(0.5,), std=(0.5,))
                                            ])
            self.dataset = datasets.ImageNet(root="ImageNet/",
                                             download=True,
                                             transform=transform)
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

if __name__ == "__main__":
    dataset = Dataset("ImageNet")
    print(len(dataset))