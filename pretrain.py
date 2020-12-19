import torch
import torch.nn as nn
from dataloader import ImageNet, DataLoader
from model.GoogLeNet import GoogLeNet, forpt

debug = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if not debug:
    assert device == 'cuda'

train_data = ImageNet(train=True)
test_data = ImageNet(train=False)
train_loader = DataLoader(train_data)
test_loader = DataLoader(test_data, batch_size=len(test_data))

model = GoogLeNet() # we should store this model
submodel = forpt()

loss = nn.CrossEntropyLoss()
opt_model = torch.optim.SGD(model.parameters(), lr=0.0001)
opt_submodel = torch.optim.SGD(submodel.parameters(), lr=0.0001)

cnt = len(train_loader)
total_epochs = 100 if not debug else 1
for epoch in range(1, total_epochs+1):
    avg_cost = 0
    for x, y in train_loader:
        y_pred = submodel(model(x))
        
        cost = loss(y_pred, y)
        opt_model.zero_grad()
        opt_submodel.zero_grad()
        cost.backward()
        opt_model.step()
        opt_submodel.step()
        
        avg_cost += cost
    avg_cost /= cnt
    print("Epoch: %d, Cost: %f"%(epoch, avg_cost))

        