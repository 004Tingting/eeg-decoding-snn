"""EEGNet and DeepConvNet, reimplemented from the papers.

- Lawhern et al., "EEGNet: a compact CNN for EEG-based brain-computer
  interfaces", J. Neural Eng. 15 (2018) 056013.
- Schirrmeister et al., "Deep learning with convolutional neural networks for
  EEG decoding and visualization", Human Brain Mapping 38 (2017) 5391-5420.
"""
import torch.nn as nn


class Conv2dWithNorm(nn.Module):
    """Conv2d with max-norm weight constraint (EEGNet paper, rho=1 on slice 2)."""
    def __init__(self, *a, max_norm=1.0, **kw):
        super().__init__()
        self.conv = nn.Conv2d(*a, **kw)
        self.max_norm = max_norm

    def forward(self, x):
        w = self.conv.weight
        self.conv.weight.data = w * torch.where(
            w.norm(p=2, dim=(2, 3), keepdim=True) > self.max_norm,
            self.max_norm / (w.norm(p=2, dim=(2, 3), keepdim=True) + 1e-12), 1.0)
        return self.conv(x)


import torch  # noqa: E402  (kept after imports-used-by-decorators for clarity)


class EEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, samples=750,
                 F1=8, D=2, kernel=64, pool1=4, pool2=8, drop=0.25):
        super().__init__()
        F2 = F1 * D
        self.temporal = nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False)
        self.depthwise = Conv2dWithNorm(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn1 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, pool1))
        self.sep_dw = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), groups=F2, bias=False)
        self.sep_pw = nn.Conv2d(F2, F2, (1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, pool2))
        self.drop = nn.Dropout(drop)
        n_out = samples
        n_out = (n_out + 2 * (kernel // 2) - kernel + 1)
        n_out //= pool1
        n_out = (n_out + 2 * 8 - 16 + 1)
        n_out //= pool2
        self.fc = nn.Linear(F2 * n_out, n_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = nn.functional.elu(self.bn1(x))
        x = self.drop(self.pool1(x))
        x = self.sep_pw(self.sep_dw(x))
        x = nn.functional.elu(self.bn2(x))
        x = self.drop(self.pool2(x))
        return self.fc(x.flatten(1))


class DeepConvNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, samples=750, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 25, (1, 10), bias=False),
            nn.Conv2d(25, 25, (n_channels, 1), groups=25, bias=False),
            nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(drop),
            nn.Conv2d(25, 50, (1, 10), bias=False),
            nn.BatchNorm2d(50), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(drop),
            nn.Conv2d(50, 100, (1, 10), bias=False),
            nn.BatchNorm2d(100), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(drop),
            nn.Conv2d(100, 200, (1, 10), bias=False),
            nn.BatchNorm2d(200), nn.ELU(), nn.MaxPool2d((1, 3)), nn.Dropout(drop),
        )
        with torch.no_grad():
            n_out = self.net(torch.zeros(1, 1, n_channels, samples)).flatten(1).shape[1]
        self.fc = nn.Linear(n_out, n_classes)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))
