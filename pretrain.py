import torch
import torch.nn as nn
from dataloader import ImageNet, DataLoader
from model.GoogLeNet import GoogLeNet, forpt
import time

debug = False

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not debug:
    assert device == 'cuda'

train_data = ImageNet(train=True)
test_data = ImageNet(train=False)
train_loader = DataLoader(train_data, batch_size=16)
test_loader = DataLoader(test_data, batch_size=len(test_data))

model = GoogLeNet().to(device) # we should store this model
submodel = forpt().to(device)

loss = nn.CrossEntropyLoss()
opt_model = torch.optim.SGD(model.parameters(), lr=0.0001)
opt_submodel = torch.optim.SGD(submodel.parameters(), lr=0.0001)

cnt = len(train_loader)
total_epochs = 100 if not debug else 1

print("Train Start!")
start = time.time()
for epoch in range(1, total_epochs+1):
    avg_cost = 0
    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        y_pred = submodel(y_pred)
        cost = loss(y_pred, y)
        opt_model.zero_grad()
        opt_submodel.zero_grad()
        cost.backward()
        opt_model.step()
        opt_submodel.step()
        
        avg_cost += cost
    avg_cost /= cnt
    print("Epoch: %d, Cost: %f, Elapsed time: %.3f"%(epoch, avg_cost, time.time()-start))

        