import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, if_norm=True, if_transpose=False,
               if_pool=False,
               device='cpu'):
    if if_transpose:
        block = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                    device=device)]
    else:
        block = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           device=device)]
    if if_norm:
        block.append(nn.InstanceNorm2d(out_channels, device=device))
    block.append(nn.GELU())
    if if_pool:
        block.append(nn.MaxPool2d(2))
    return nn.Sequential(*block)


class Residual2DBlock(nn.Module):
    def __init__(self, in_channel, times_channel=4, kernel_size=3, stride=1, padding=1, device='cpu'):
        super(Residual2DBlock, self).__init__()
        self.in_channel = in_channel
        self.times_channel = times_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv_layer0 = nn.Conv2d(in_channel, in_channel * times_channel, kernel_size, stride, padding,
                                     device=device)
        self.norm0 = nn.InstanceNorm2d(in_channel * times_channel)
        self.conv_layer1 = nn.Conv2d(in_channel * times_channel, in_channel, kernel_size, stride, padding,
                                     device=device)
        self.norm1 = nn.InstanceNorm2d(in_channel)

    def forward(self, x):
        res = self.conv_layer0(x)
        res = self.norm0(res)
        res = F.gelu(res)

        res = self.conv_layer1(res)
        res = self.norm1(res)

        x = torch.add(x, res)
        x = F.gelu(x)
        return x


class SimpleGenerator(nn.Module):
    '''
    dla 64x64
    gen_dims = [256, 128, 64, 32]
    gen_kernel_sizes = [4, 4, 4, 8]
    gen_strides = [2, 2, 2, 4]
    gen_paddings = [0, 1, 1, 2]
    '''

    def __init__(self, latent_dim, dims, kernel_sizes, strides, paddings, new_size, if_res=True, device='cpu'):
        super(SimpleGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        if len(new_size) < 2:
            new_size = (new_size, new_size)
        self.new_size = new_size
        self.n_times = len(kernel_sizes)
        self.dense_init = nn.Linear(latent_dim, dims[0], device=device)
        if if_res:
            if_res = [True for _ in range(self.n_times)]
        blocks = []
        for i in range(self.n_times):
            blocks.append(conv_block(dims[i],
                                     dims[i + 1],
                                     kernel_size=kernel_sizes[i],
                                     stride=strides[i],
                                     padding=paddings[i],
                                     if_transpose=True,
                                     device=device))
            if if_res[i]:
                blocks.append(Residual2DBlock(dims[i + 1], device=device))

        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(dims[-1], 3, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = self.dense_init(x)
        x = F.gelu(x)
        x = x.view(len(x), -1, 1, 1)
        x = self.blocks(x)
        x = self.out_conv(x)
        x = F.tanh(x)
        x = v2.Resize(self.new_size)(x)

        return x.to(self.device)


class SimpleDiscriminator(nn.Module):
    '''
    dla 64x64
    disc_dims = [64, 128, 256]
    disc_kernel_sizes = [4, 4, 4, 4]
    disc_strides = [2, 2, 2, 4]
    disc_paddings = [1, 1, 1, 1]
    disc_if_pools = [True, False, True, False]
    '''

    def __init__(self, dims, kernel_sizes, strides, paddings, if_pools, if_res=True, device='cpu'):
        super(SimpleDiscriminator, self).__init__()
        self.input_norm = nn.InstanceNorm2d(3, device=device)
        self.device = device
        self.n_times = len(kernel_sizes)
        blocks = []
        dims = [3, ] + dims
        if_norm = True
        if if_res:
            if_res = [True for _ in range(self.n_times)]
        for i in range(self.n_times):
            if i == (self.n_times - 1):
                if_norm = False
            blocks.append(conv_block(dims[i],
                                     dims[i + 1],
                                     kernel_size=kernel_sizes[i],
                                     stride=strides[i],
                                     padding=paddings[i],
                                     if_norm=if_norm,
                                     if_pool=if_pools[i],
                                     device=device))
            if if_res[i] & if_norm:
                blocks.append(Residual2DBlock(dims[i + 1], device=device))
        self.block = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(dims[-1], 1024, device=device)  ## tu dodawać
        self.out_dense = nn.Linear(1024, 1, device=device)  ## tu dodawać

    def forward(self, x):
        x = self.input_norm(x)
        x = self.block(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.out_dense(x)
        return x.to(self.device)
