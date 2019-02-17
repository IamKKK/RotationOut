import torch.nn as nn

from .tools import Dropout, Rotation

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride, bn_weight, drop_rate, theta):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if theta == 'dropout':
            self.drop1 = Dropout(drop_rate) if planes > 32 else nn.Sequential()
            self.drop2 = Dropout(drop_rate) if planes > 32 else nn.Sequential()
        else:
            self.drop1 = Rotation(in_planes, drop_rate, theta=theta) if planes > 32 else nn.Sequential()
            self.drop2 = Rotation(planes, drop_rate, theta=theta) if planes > 32 else nn.Sequential()
            
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
            nn.init.kaiming_normal_(self.shortcut[0].weight)
            nn.init.constant_(self.shortcut[1].weight, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)

        nn.init.constant_(self.bn2.weight, bn_weight)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        out = self.drop1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.drop2(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(x)
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, num_layers=110, num_classes=100, block=BasicBlock, drop_rate=0.0, theta='dropout'):
        super(ResNet, self).__init__()
        self.in_planes = 16
        num_blocks = (num_layers-2)//6
        self.bn_weight = num_blocks ** -.5
        
        self.drop_rate = drop_rate
        self.theta = theta

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks, 1)
        self.layer2 = self._make_layer(block, 32, num_blocks, 2)
        self.layer3 = self._make_layer(block, 64, num_blocks, 2)

        self.linear = nn.Linear(64, num_classes)

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.bn1.weight, 1.0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride,
                bn_weight=self.bn_weight, drop_rate=self.drop_rate, theta=self.theta))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.mean((2,3))
        out = self.linear(out)
        return out