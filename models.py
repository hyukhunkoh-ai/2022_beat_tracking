import torch
import torch.nn as nn
from argparse import ArgumentParser
'''
>>> torch.Size([32])
    # 1d: [batch_size] 
    # use for target labels or predictions.
>>> torch.Size([12, 256])
    # 2d: [batch_size, num_features (aka: C * H * W)]
    # use for nn.Linear() input.
>>> torch.Size([10, 1, 2048])
    # 3d: [batch_size, channels, num_features (aka: H * W)]
    # when used as nn.Conv1d() input.
    # (but [seq_len, batch_size, num_features]
    # if feeding an RNN).
>>> torch.Size([16, 3, 28, 28])
    # 4d: [batch_size, channels, height, width]
    # use for nn.Conv2d() input.
>>>  torch.Size([32, 1, 5, 15, 15])
    # 5D: [batch_size, channels, depth, height, width]
    # use for nn.Conv3d() input.


'''




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


class TcnModel(nn.Module):
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
        super(TcnModel, self).__init__()

        self.blocks = torch.nn.ModuleList()
        for n in range(nblocks):
            in_ch = ninputs if n == 0 else out_ch
            out_ch = channel_width if n == 0 else in_ch + channel_width
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
            # print(x.shape)
            x = block(x)


        # output == None,256,1103 if input == None,1,282240(12.8 second)
        return x

class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=2, num_classes=2, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 1)

        batch_size, length, channels = out1.shape

        out2 = out1.view(batch_size, length, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)

class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=2, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv1d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv1d(feature_size, num_anchors * 2, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        out = out.permute(0, 2, 1)

        return out.contiguous().view(out.shape[0], -1, 2)
