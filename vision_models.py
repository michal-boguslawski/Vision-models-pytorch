import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2


def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1, if_transpose=False, if_pool=False,
               device='cpu'):
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


def res_net_block():
    None


class SimpleGenerator(nn.Module):
    '''
    dla 64x64
    gen_dims = [256, 128, 64, 32]
    gen_kernel_sizes = [4, 4, 4, 8]
    gen_strides = [2, 2, 2, 4]
    gen_paddings = [0, 1, 1, 2]
    '''
    def __init__(self, latent_dim, dims, kernel_sizes, strides, paddings, new_size, device='cpu'):
        super(SimpleGenerator, self).__init__()
        self.latent_dim = latent_dim
        self.device = device
        if len(new_size) < 2:
            new_size = (new_size, new_size)
        self.new_size = new_size
        self.n_times = len(kernel_sizes)
        self.dense_init = nn.Linear(latent_dim, dims[0])
        blocks = []
        for i in range(self.n_times):
            blocks.append(conv_block(dims[i],
                                     dims[i+1],
                                     kernel_size=kernel_sizes[i],
                                     stride=strides[i],
                                     padding=paddings[i],
                                     if_transpose=True,
                                     device=device))
        self.blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(dims[-1], 3, kernel_size=3, stride=1, padding=1, device=device)

    def forward(self, x):
        x = self.dense_init(x)
        x = F.gelu(x)
        x = x.view(-1, self.latent_dim, 1, 1)
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
    def __init__(self, dims, kernel_sizes, strides, paddings, if_pools, device='cpu'):
        super(SimpleDiscriminator, self).__init__()
        self.input_norm = nn.InstanceNorm2d(3, device=device)
        self.device = device
        self.n_times = len(kernel_sizes)
        blocks = []
        dims = [3, ] + dims
        for i in range(self.n_times-1):
            blocks.append(conv_block(dims[i],
                                     dims[i+1],
                                     kernel_size=kernel_sizes[i],
                                     stride=strides[i],
                                     padding=paddings[i],
                                     if_pool=if_pools[i],
                                     device=device))
        self.block = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(dims[-1], 1,
                                  kernel_size=kernel_sizes[-1],
                                  stride=strides[-1],
                                  padding=paddings[-1],
                                  device=device)
        self.flatten = nn.Flatten()
        #self.dense = nn.Linear() ## tu dodawać

    def forward(self, x):
        x = self.input_norm(x)
        x = self.block(x)
        x = self.out_conv(x)
        x = x.view(-1, 1)
        return x.to(self.device)
