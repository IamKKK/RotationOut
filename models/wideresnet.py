import torch.nn as nn

from .tools import Dropout, Rotation

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate, theta):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(in_planes)
        self.bn2 = nn.BatchNorm2d(out_planes)

        self.relu = nn.ReLU(inplace=True)
        if theta == 'dropout':
            self.drop1 = Dropout(drop_rate) if out_planes > 160 else nn.Sequential()
            self.drop2 = Dropout(drop_rate) if out_planes > 160 else nn.Sequential()
        else:
            self.drop1 = Rotation(in_planes, drop_rate, theta=theta) if out_planes > 160 else nn.Sequential()
            self.drop2 = Rotation(out_planes, drop_rate, theta=theta) if out_planes > 160 else nn.Sequential()

        self.convShortcut = nn.Sequential()
        if in_planes != out_planes or stride!=1:
            self.convShortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        if not self.convShortcut:
            res = x
        x = self.bn1(x)
        x = self.relu(x)
        if self.convShortcut:
            res = self.convShortcut(x)

        out = self.drop1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.drop2(out)
        out = self.conv2(out)

        out += res
        return out

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=100, widen_factor=10, drop_rate=0.3, theta='dropout'):
        super(WideResNet, self).__init__()
        num_blocks = (depth - 4) // 6
        self.drop_rate = drop_rate
        self.theta = theta

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.block1 = self._make_layer(num_blocks, 16, 16*widen_factor, BasicBlock, 1)
        self.block2 = self._make_layer(num_blocks, 16*widen_factor, 32*widen_factor, BasicBlock, 2)
        self.block3 = self._make_layer(num_blocks, 32*widen_factor, 64*widen_factor, BasicBlock, 2)

        self.bn1 = nn.BatchNorm2d(64*widen_factor)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64*widen_factor, num_classes)


        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn1.weight, 1.0)

    def _make_layer(self, num_blocks, in_planes, out_planes, block, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        planes = in_planes
        for stride in strides:
            layers.append(block(planes, out_planes, stride, self.drop_rate, self.theta))
            planes = out_planes
        return nn.Sequential(*layers)


    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = out.mean((2,3))
        return self.fc(out)