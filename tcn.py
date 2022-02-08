import torch
from argparse import ArgumentParser

def get_activation(act_type,ch=None):
    """ Helper function to construct activation functions by a string.
    Args:
        act_type (str): One of 'ReLU', 'PReLU', 'SELU', 'ELU'.
        ch (int, optional): Number of channels to use for PReLU.

    Returns:
        torch.nn.Module activation function.
    """

    if act_type == "PReLU":
        return torch.nn.PReLU(ch)
    elif act_type == "ReLU":
        return torch.nn.ReLU()
    elif act_type == "SELU":
        return torch.nn.SELU()
    elif act_type == "ELU":
        return torch.nn.ELU()


class dsTCNBlock(torch.nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 norm_type="BatchNorm",
                 act_type="PReLU"):
        super(dsTCNBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.stride = stride
        self.norm_type = norm_type

        pad_value = ((kernel_size - 1) * dilation) // 2

        self.conv1 = torch.nn.Conv1d(in_ch,
                                     out_ch,
                                     kernel_size=kernel_size,
                                     stride=stride,
                                     dilation=dilation,
                                     padding=pad_value)
        self.act1 = get_activation(act_type, out_ch)

        if norm_type == "BatchNorm":
            self.norm1 = torch.nn.BatchNorm1d(out_ch)
            self.res_norm = torch.nn.BatchNorm1d(out_ch)
        else:
            self.norm1 = None
            self.res_norm = None

        self.res_conv = torch.nn.Conv1d(in_ch,
                                        out_ch,
                                        kernel_size=1,
                                        stride=stride)

    def forward(self, x):
        x_res = x

        x = self.conv1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.act1(x)

        # -- residual connection --
        x_res = self.res_conv(x_res)
        if self.res_norm is not None:
            x_res = self.res_norm(x_res)

        return x + x_res


class dsTCNModel():
    """ Downsampling Temporal convolutional network.
        Args:
            ninputs (int): Number of input channels (mono = 1, stereo 2). Default: 1
            noutputs (int): Number of output channels (mono = 1, stereo 2). Default: 1
            nblocks (int): Number of total TCN blocks. Default: 10
            kernel_size (int): Width of the convolutional kernels. Default: 15
            stride (int): Stide size when applying convolutional filter. Default: 2 
            dialation_growth (int): Compute the dilation factor at each block as dilation_growth ** (n % stack_size). Default: 8
            channel_growth (int): Compute the output channels at each black as in_ch * channel_growth. Default: 1
            channel_width (int): When channel_growth = 1 all blocks use convolutions with this many channels. Default: 32
            stack_size (int): Number of blocks that constitute a single stack of blocks. Default: 10
            norm_type (str): Type of normalization layer to use 'BatchNorm'
    """

    def __init__(self,
                 ninputs=1,
                 noutputs=2,
                 nblocks=8,
                 kernel_size=15,
                 stride=2,
                 dilation_growth=8,
                 channel_growth=1,
                 channel_width=32,
                 stack_size=4,
                 norm_type='BatchNorm',
                 act_type='PReLU',
                 **kwargs):
        super(dsTCNModel, self).__init__()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = ninputs if n == 0 else out_ch
            out_ch = channel_width if n == 0 else in_ch + channel_growth
            dilation = dilation_growth ** (n % stack_size)

            self.blocks.append(dsTCNBlock(
                in_ch,
                out_ch,
                kernel_size,
                stride,
                dilation,
                norm_type,
                act_type
            ))

    def forward(self, x):

        for block in self.blocks:
            x = block(x)



        return x
