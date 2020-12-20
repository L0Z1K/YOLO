import torch
import torch.nn as nn
from dataloader import ImageNet, DataLoader
from model.GoogLeNet import forpt
import time

debug = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not debug:
    assert device == 'cuda'
    '''not training on cuda! check GPU state'''

if __name__ == "__main__":
    train_data = ImageNet(train=True)
    test_data = ImageNet(train=False)
    train_loader = DataLoader(train_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=len(test_data))

    pretrain_model = forpt().to(device)

    loss = nn.CrossEntropyLoss()
    opt_pretrain_model = torch.optim.SGD(pretrain_model.parameters(), lr=0.0001)

    cnt = len(train_loader)
    total_epochs = 100 if not debug else 1

    print("Train Start!")
    start = time.time()
    for epoch in range(1, total_epochs+1):
        avg_cost = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            y_pred = pretrain_model(x)
            cost = loss(y_pred, y)
            opt_pretrain_model.zero_grad()
            cost.backward()
            pretrain_model.step()
            
            avg_cost += cost
        avg_cost /= cnt
        print("Epoch: %d, Cost: %f, Elapsed time: %.3f"%(epoch, avg_cost, time.time()-start))
        if epoch % 10 == 0:
            torch.save(pretrain_model, f'pretrain_{epoch}.pt')
    torch.save(pretrain_model, 'pretrain_final.pt')