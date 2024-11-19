import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, if_transpose=False, if_pool=False,
               device='cuda'):
    if if_transpose:
        block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    device=device)]
    else:
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           device=device)]
    block.append(nn.InstanceNorm2d(out_channels, device=device))
    block.append(nn.GELU())
    if if_pool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)


class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, dim, n_times, device='cuda'):
        super(SimpleGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.input_norm = nn.LayerNorm(latent_dim, device=device)
        self.n_times = n_times
        blocks = [conv_block(latent_dim, dim, kernel_size=4, stride=2, padding=0, if_transpose=True)]
        for i in range(n_times):
            blocks.append(
                conv_block(dim // 2 ** i, dim // 2 ** (i + 1), kernel_size=4, stride=2, padding=1, if_transpose=True))
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(dim // 2 ** n_times, 3, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = x.view(-1, self.latent_dim, 1, 1)
        x = self.blocks(x)
        x = self.out_conv(x)
        return F.tanh(x).to(self.device)


class SimpleDiscriminator(nn.Module):
    def __init__(self, dim, n_times, device='cuda'):
        super(SimpleDiscriminator, self).__init__()
        self.input_norm = nn.InstanceNorm2d(3, device=device)
        self.device = device
        self.n_times = n_times
        block = conv_block(3, dim, kernel_size=4, stride=2, padding=1, if_pool=False)
        for i in range(n_times):
            block.append(
                conv_block(dim * 2 ** i, dim * 2 ** (i + 1), kernel_size=4, stride=2, padding=1, if_pool=False))
        self.block = nn.Sequential(*block)
        self.out_conv = nn.Conv2d(dim * 2 ** n_times, 1, kernel_size=4, stride=1, padding=0, device=device)

    def forward(self, x):
        x = self.input_norm(x)
        x = self.block(x)
        x = self.out_conv(x)
        x = x.view(-1, 1)
        return x.to(self.device)
