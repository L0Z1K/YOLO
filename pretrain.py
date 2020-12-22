import time
import datetime
import torch
import torch.nn as nn
from dataloader import ImageNet, DataLoader
from model.GoogLeNet import forpt

debug = False
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LOAD_PATH = '/content/gdrive/My Drive/YOLO/pretrain_10.pt'
SAVE_PATH = '/content/gdrive/My Drive/YOLO/'

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
    torch.save({
      'epoch': epoch,
      'model_state_dict': pretrain_model.state_dict(),
      'optimizer_state_dict': opt_pretrain_model.state_dict(),
      'loss': avg_cost,
    }, f'{SAVE_PATH}pretrain_final.pt')
