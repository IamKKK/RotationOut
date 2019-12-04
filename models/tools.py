import torch
import torch.nn as nn
import numpy as np


class Dropout(nn.Module):
    def __init__(self, drop_rate, center=False):
        super(Dropout, self).__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.center = center

    def forward(self, x):
        if not self.training: 
            return x
        if self.center:
            c = x.mean(dim=(0,2,3), keepdim=True)
            return self.drop(x-c) + c
        else:
            return self.drop(x)

class Rotation(nn.Module):
    def __init__(self, dim, drop_rate=0.5, 
                    device='cuda', pre_loop=20000, theta='Gaussian'):
        super(Rotation, self).__init__()
        self.dim = dim
        self.device = device
        self.pre_loop = pre_loop

        self.theta = theta
        if self.theta == 'Gaussian':
            drop_std = (drop_rate/(1-drop_rate))**0.5
        elif self.theta == 'Uniform':
            drop_std = (3*drop_rate/(1-drop_rate))**0.5
        self.drop_std = torch.tensor(drop_std, dtype=torch.float32, device=device)


        self.indexes = self.generate_index(dim)
        self.count = 0

    def random_permutation(self, dim):
        shuffle = np.random.permutation(dim)
        index1 = np.zeros(dim, dtype=np.int64)
        index2 = np.ones((1,dim,1,1), dtype=np.float32)
        for item in range(0, dim, 2):
            index1[shuffle[item]] = shuffle[item+1]
            index1[shuffle[item+1]] = shuffle[item]
            index2[0, shuffle[item]] = -1
        return index1, index2

    def generate_index(self, dim):
        indexes = [self.random_permutation(dim) for i in range(self.pre_loop)]
        indexes1 = torch.tensor(np.array([i[0] for i in indexes])).to(self.device)
        indexes2 = torch.tensor(np.array([i[1] for i in indexes])).to(self.device)
        indexes = [(indexes1[i], indexes2[i]) for i in range(self.pre_loop)]
        del indexes1, indexes2
        return indexes

    def forward(self, x):
        if not self.training: return x
        if self.count < 1000:
            drop_std = self.drop_std * self.count / 1000.
        else:
            drop_std = self.drop_std
        bs = x.shape[0]
        if self.theta == 'Gaussian':
            theta = drop_std * torch.randn(size=(bs,1,1,1), device=self.device)
        elif self.theta == 'Uniform':
            theta = drop_std * torch.rand(size=(bs,1,1,1), device=self.device)

        index1, index2 = self.indexes[self.count%self.pre_loop]
        self.count += 1

        noise = (x-x.mean(dim=(0,2,3), keepdim=True)).index_select(1, index1) * index2 * theta
        x = x + noise
        return x
