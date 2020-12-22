import time
import datetime
import torch
import torch.nn as nn
from dataloader import ImageNet, DataLoader
from model.GoogLeNet import forpt
import numpy as np

debug = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_PATH = '/content/gdrive/My Drive/YOLO/pretrain_40.pt'
SAVE_PATH = '/content/gdrive/My Drive/YOLO/'

if not debug:
    assert device == 'cuda'
    '''not training on cuda! check GPU state'''

def correct(output, target, topk=[1]):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        correct = pred.eq(target.view(-1, 1).expand(-1, maxk))

        res = []
        for k in topk:
            correct_k = correct[:,:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return np.array(res, dtype = float)

def test(test_loader, pretrain_model, topk = [1, 5]):
    pretrain_model.eval()
    res = np.zeros(len(topk), dtype=float)
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        y_pred = pretrain_model(x)
        res += correct(y_pred, y, topk)
    res = res / len(test_loader.dataset) * 100
    return topk, res


def train(epoch, train_loader, pretrain_model, opt_pretrain_model):
    pretrain_model.train()
    avg_cost = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = pretrain_model(x)
        cost = loss(y_pred, y)
        opt_pretrain_model.zero_grad()
        cost.backward()
        opt_pretrain_model.step()
        
        avg_cost += cost
    avg_cost /= cnt
    print(f"Epoch: {epoch}, Cost: {avg_cost}, Elapsed time:{str(datetime.timedelta(seconds=time.time()-start))}")
    if epoch % 10 == 0:
        torch.save({
          'epoch': epoch,
          'model_state_dict': pretrain_model.state_dict(),
          'optimizer_state_dict': opt_pretrain_model.state_dict(),
          'loss': avg_cost,
        }, f'{SAVE_PATH}pretrain_{epoch}.pt')


if __name__ == "__main__":
    train_data = ImageNet(train=True)
    test_data = ImageNet(train=False)
    train_loader = DataLoader(train_data, batch_size=32)
    test_loader = DataLoader(test_data, batch_size=32)

    pretrain_model = forpt().to(device)

    loss = nn.CrossEntropyLoss()
    opt_pretrain_model = torch.optim.SGD(pretrain_model.parameters(), lr=0.0001)
    
    if LOAD_PATH is not '':
        checkpoint = torch.load(LOAD_PATH)
        pretrain_model.load_state_dict(checkpoint['model_state_dict'])
        opt_pretrain_model.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'saved model info:\nstart_epoch: {start_epoch}\nloss: {checkpoint["loss"]}')
    else:
        start_epoch = 1

    cnt = len(train_loader)
    total_epoch = 100 if not debug else 1

    print(f'Train {"start" if start_epoch is 1 else "retstart"}! on {device}{"with debug on" if debug else ""}')

    start = time.time()
    for epoch in range(start_epoch, total_epoch+1):
        train(epoch, train_loader, pretrain_model, opt_pretrain_model)
        topk, accuracies = test(test_loader, pretrain_model)
        print(f'test result at {epoch}')
        for k, accuracy in zip(topk, accuracies):
            print(f'top {k} accuracy: {accuracy}')
        
    torch.save({
      'epoch': epoch,
      'model_state_dict': pretrain_model.state_dict(),
      'optimizer_state_dict': opt_pretrain_model.state_dict(),
    }, f'{SAVE_PATH}pretrain_final.pt')
