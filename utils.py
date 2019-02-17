import torch, torchvision
import torchvision.transforms as transforms


def train(net, trainloader, criterion, optimizer):
    net.train()
    train_loss, correct, total = 0., 0., 0.
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return train_loss/(batch_idx+1), 100.*correct/total


def test(net, testloader, criterion):
    net.eval()
    test_loss, correct, total = 0., 0., 0.
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss/(batch_idx+1), 100.*correct/total

def cifar_loader():
    # cifar10_mu = (0.4246676 , 0.41490164, 0.38390577)
    # cifar10_std = (0.2828169 , 0.27786118, 0.28444958)

    cifar100_mu = (0.4914, 0.4822, 0.4465)
    cifar100_std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mu, cifar100_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mu, cifar100_std),
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform_train, download=True)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,  transform=transform_test, download=True)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
    return trainloader, testloader