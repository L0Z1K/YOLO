import argparse
import torch.utils.data as data
from torchvision import datasets, transforms

def ImageNet(train=True, image_size=(448,448)):
    return datasets.ImageFolder(root="imagenet-mini/train" if train else "imagenet-mini/val",
                                transform=transforms.Compose([
                                transforms.RandomVerticalFlip(p=0.5),
                                transforms.Resize(image_size),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                                ]))

def DataLoader(dataset, batch_size=128):
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           drop_last=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', type=int, default=128, help="Batch Size (default: 128)")
    parser.add_argument('-t', type=bool, default=True, help="Train data if True, else Test data (default: True)")
    parser.add_argument('-s', type=tuple, default=(448,448), help="Image Size (default: (448, 448))")
    
    parser.print_help()
    args = parser.parse_args()

    imagenet = ImageNet(train=args.t, image_size=args.s)
    data_loader = DataLoader(imagenet, batch_size=args.b)

    for data, label in data_loader:
        print(data.shape)
        print(label.shape)
        break