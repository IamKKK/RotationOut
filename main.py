import torch, sys, time, argparse
torch.backends.cudnn.benchmark = True

from models import *
from utils import train, test, cifar_loader


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str)
parser.add_argument('--drop_rate', type=float)
parser.add_argument('--drop_type', type=str)
args = parser.parse_args()

model_name = args.model_name
drop_rate = args.drop_rate
drop_type = args.drop_type

print('Training {} with drop rate: {}, drop_type: "{}".'.format(model_name, drop_rate, drop_type))


if model_name == 'resnet':
    net = ResNet(drop_rate=drop_rate, theta=drop_type)
elif model_name == 'wideresnet':
    net = WideResNet(drop_rate=drop_rate, theta=drop_type)



net = net.to('cuda')
trainloader, testloader = cifar_loader()
criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[82, 123, 160], gamma=0.1)

for epoch in range(165):
    t = time.time()
    scheduler.step()
    train_loss, train_acc = train(net, trainloader, criterion, optimizer)
    test_loss, test_acc = test(net, testloader, criterion)
    print('Epoch %d: train loss %.3f, acc: %.3f; test loss: %.3f, acc %.3f; time: %.3f min'%
            (epoch, train_loss, train_acc, test_loss, test_acc, (time.time()-t)/60))

print('Training {} with drop rate: {}, drop_type: "{}" done.'.format(model_name, drop_rate, drop_type))
