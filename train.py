import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 


def train_iter(train_loader, model, optim, loss, device, loss_val = []):
    samples = len(train_loader.dataset)
    model.train()
    model.to(device)
    optim.zero_grad()
    step = len(train_loader)//5
    for i, (data, target) in enumerate(train_loader):
        target = target.squeeze(dim = 1)
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optim.step()
        optim.zero_grad()
        if i % step == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(train_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())



def evaluate(val_loader, model, loss_val):
    model.eval()
    
    samples = len(val_loader.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data,target in val_loader:
            target = target.squeeze(dim=1)
            output = F.log_softmax(model(data.cuda()), dim=1)
            loss = F.nll_loss(output, target.cuda(), reduction='sum')
            _, pred = torch.max(output, dim=1)
            
            tloss += loss.item()
            csamp += pred.eq(target.cuda()).sum()
            
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
