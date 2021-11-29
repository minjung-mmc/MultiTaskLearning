import torch
import torch.nn as nn
from torch.nn.modules import activation


class Bottleneck(nn.Module):
    def __init__(
        self,
        ch,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        dropout_p=0,
        bias=False,
        relu=False,
    ):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > ch:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}.".format(ch, internal_ratio)
            )

        internal_ch = ch // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # projection
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(ch, internal_ch, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_ch),
            activation(),
        )

        # asymmetric:
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_ch,
                    internal_ch,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_ch),
                activation(),
                nn.Conv2d(
                    internal_ch,
                    internal_ch,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_ch),
                activation(),
            )
        # regular
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_ch,
                    internal_ch,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_ch),
                activation(),
            )

        # expansion
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_ch, ch, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(ch),
            activation(),
        )

        self.ext_regularizer = nn.Dropout2d(p=dropout_p)

        self.out_activation = activation()

    def forward(self, x):
        main = x

        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regularizer(ext)

        out = main + ext
        out = self.out_activation(out)

        return out


class DownsamplingBottleneck(nn.Module):
    def __init__(
        self,
        input_ch,
        output_ch,
        internal_ratio=4,
        retrun_indices=False,
        dropout_p=0,
        bias=False,
        relu=False,
    ):
        super().__init__()

        self.retrun_indices = retrun_indices

        if internal_ratio <= 1 or internal_ratio > input_ch:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}. ".format(
                    input_ch, internal_ratio
                )
            )

        internal_ch = input_ch // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_max1 = nn.MaxPool2d(2, stride=2, return_indices=retrun_indices)

        # in downsampling, projection replaced with a 2X2 convolution with stride 2
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(input_ch, internal_ch, kernel_size=2, stride=2, bias=bias),
            nn.BatchNorm2d(internal_ch),
            activation(),
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_ch, internal_ch, kernel_size=3, stride=1, padding=1, bias=bias
            ),
            nn.BatchNorm2d(internal_ch),
            activation(),
        )

        # expans
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_ch, output_ch, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(output_ch),
            activation(),
        )

        self.ext_regularizer = nn.Dropout2d(p=dropout_p)

        self.out_activation = activation()

    def forward(self, x):
        # Main
        if self.retrun_indices:
            main, max_ind = self.main_max1(x)

        else:
            main = self.main_max1(x)

        # Extension
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regularizer(ext)

        # add zero pad, to match the num of feature map
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        if main.is_cuda:
            padding = padding.to("cuda:2")

        main = torch.cat((main, padding), 1)

        out = main + ext
        out = self.out_activation(out)

        return out, max_ind


class UpsamplingBottleneck(nn.Module):
    def __init__(
        self, input_ch, output_ch, internal_ratio=4, dropout_p=0, bias=False, relu=False
    ):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > input_ch:
            raise RuntimeError(
                "Value out of range. Expected value in the "
                "interval [1, {0}], got internal_scale={1}. ".format(
                    input_ch, internal_ratio
                )
            )

        internal_ch = input_ch // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(input_ch, output_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(output_ch),
        )

        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # projection
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(input_ch, internal_ch, kernel_size=1, stride=1, bias=bias),
            nn.BatchNorm2d(internal_ch),
        )

        # Transposed convolution
        self.ext_tconv1 = nn.ConvTranspose2d(
            internal_ch, internal_ch, kernel_size=2, stride=2, bias=bias
        )
        self.ext_tconv1_bnorm = nn.BatchNorm2d(internal_ch)
        self.ext_tconv1_activation = activation()

        # expansion
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_ch, output_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(output_ch),
        )

        self.ext_regularizer = nn.Dropout2d(p=dropout_p)

        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # Main
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices, output_size=output_size)

        # extension
        ext = self.ext_conv1(x)
        ext = self.ext_tconv1(ext, output_size=output_size)
        ext = self.ext_tconv1_bnorm(ext)
        ext = self.ext_tconv1_activation(ext)
        ext = self.ext_conv2(ext)
        ext = self.ext_regularizer(ext)

        out = main + ext
        out = self.out_activation(out)

        return out
