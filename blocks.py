import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * 255 * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class BaseConv(nn.Module):
    def __init__(
        self, n_feats, ksize, depthwise_sep=False):
        super().__init__()

        if depthwise_sep:
            layers = [nn.Conv2d(n_feats, n_feats, ksize, padding=ksize//2, groups=n_feats, bias=False),
                nn.PReLU(),
                nn.Conv2d(n_feats, n_feats, 1, bias=True),
                nn.PReLU()]
        else:
            layers = [nn.Conv2d(n_feats, n_feats, ksize, padding=ksize//2, bias=True),
                nn.PReLU()]

        self.model = nn.Sequential(*layers)
    
    def forward(self, input):
        return self.model(input)

class Upsamplingx2(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
        self.extend = nn.Conv2d(n_feats, 12, 3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
    
    def forward(self, input):
        return self.shuffle(self.extend(input))
