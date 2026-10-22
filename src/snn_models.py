"""Spiking (LIF) EEG decoder — the neuromorphic extension of the EEG project.

A spiking convolutional network in the EEGNet mould, trained with surrogate
gradients (pure PyTorch, no external SNN framework):

  analog front end : Conv2d(1->F1, (1,64)) + BN        (applied once)
  spiking stages   : per timestep t = 1..T, the SAME analog feature map drives
                     a conv -> LIF stack; membranes integrate across steps,
                     spikes use a sigmoid surrogate for backward.
  readout          : spike-count features averaged over T -> linear.

Design intent: spike-domain compute is the coding style of memristive
neuromorphic hardware — this model is the bridge between the EEG decoding
project and the memristive in-memory computing project (same repository
family, same BCI-IV 2a protocol, directly comparable numbers).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=2.0):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        sg = torch.sigmoid(ctx.alpha * x)
        return g * ctx.alpha * sg * (1 - sg), None


def spike(x):
    return SurrogateSpike.apply(x)


class LIF(nn.Module):
    """Leaky integrate-and-fire neuron with hard reset."""

    def __init__(self, decay=0.9, thresh=1.0):
        super().__init__()
        self.decay, self.thresh = decay, thresh
        self.mem = None

    def forward(self, cur):
        if self.mem is None or self.mem.shape != cur.shape:
            self.mem = torch.zeros_like(cur)
        self.mem = self.mem * self.decay + cur
        spk = spike(self.mem - self.thresh)
        self.mem = self.mem - spk * self.thresh
        return spk

    def reset(self):
        self.mem = None


class SpikingEEGNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, samples=750,
                 F1=8, D=2, kernel=64, T=8, drop=0.2):
        super().__init__()
        F2 = F1 * D
        self.T = T
        self.analog = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        self.conv_dw = nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False)
        self.bn1 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.conv_sep = nn.Conv2d(F2, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop = nn.Dropout(drop)
        self.lif1, self.lif2, self.lif3 = LIF(), LIF(), LIF()
        n_out = (samples + 2 * (kernel // 2) - kernel + 1) // 4
        n_out = (n_out + 2 * 8 - 16 + 1) // 8
        self.fc = nn.Linear(F2 * n_out, n_classes)

    def _step(self, x):
        s1 = self.lif1(self.bn1(self.conv_dw(x)))
        s1 = self.drop(self.pool1(s1))
        s2 = self.lif2(self.bn2(self.conv_sep(s1)))
        s2 = self.drop(self.pool2(s2))
        return self.lif3(s2)

    def forward(self, x):
        feat = self.analog(x)
        for m in (self.lif1, self.lif2, self.lif3):
            m.reset()
        acc = 0.0
        for _ in range(self.T):
            acc = acc + self._step(feat)
        return self.fc((acc / self.T).flatten(1))
