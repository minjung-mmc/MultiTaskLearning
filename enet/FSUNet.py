import torch
import torch.nn as nn
from torch.nn.modules import padding

from InitialBlock import *
from Bottleneck import *


class FSUNet(nn.Module):
    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Shared Encoder

        # Stage 1
        self.downsample1_0 = DownsamplingBottleneck(
            16, 64, retrun_indices=True, dropout_p=0.01, relu=encoder_relu
        )
        self.regular1_1 = Bottleneck(64, padding=1, dropout_p=0.01, relu=encoder_relu)
        self.regular1_2 = Bottleneck(64, padding=1, dropout_p=0.01, relu=encoder_relu)
        self.regular1_3 = Bottleneck(64, padding=1, dropout_p=0.01, relu=encoder_relu)
        self.regular1_4 = Bottleneck(64, padding=1, dropout_p=0.01, relu=encoder_relu)

        # Stage 2
        self.downsample2_0 = DownsamplingBottleneck(
            64, 128, retrun_indices=True, dropout_p=0.1, relu=encoder_relu
        )
        self.regular2_1 = Bottleneck(128, padding=1, dropout_p=0.1, relu=encoder_relu)
        self.dilated2_2 = Bottleneck(
            128, dilation=2, padding=2, dropout_p=0.1, relu=encoder_relu
        )
        self.asymmetric2_3 = Bottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_p=0.1,
            relu=encoder_relu,
        )
        self.dilated2_4 = Bottleneck(
            128, dilation=4, padding=4, dropout_p=0.1, relu=encoder_relu
        )
        self.regular2_5 = Bottleneck(128, padding=1, dropout_p=0.1, relu=encoder_relu)
        self.dilated2_6 = Bottleneck(
            128, dilation=8, padding=8, dropout_p=0.1, relu=encoder_relu
        )
        self.asymmetric2_7 = Bottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_p=0.1,
            relu=encoder_relu,
        )
        self.dilated2_8 = Bottleneck(
            128, dilation=16, padding=16, dropout_p=0.1, relu=encoder_relu
        )

        # Decoder

        # Stage 3
        self.regular3_0 = Bottleneck(128, padding=1, dropout_p=0.1, relu=decoder_relu)
        self.dilated3_1 = Bottleneck(
            128, dilation=2, padding=2, dropout_p=0.1, relu=decoder_relu
        )
        self.asymmetric3_2 = Bottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_p=0.1,
            relu=decoder_relu,
        )
        self.dilated3_3 = Bottleneck(
            128, dilation=4, padding=4, dropout_p=0.1, relu=decoder_relu
        )
        self.regular3_4 = Bottleneck(128, padding=1, dropout_p=0.1, relu=decoder_relu)
        self.dilated3_5 = Bottleneck(
            128, dilation=8, padding=8, dropout_p=0.1, relu=decoder_relu
        )
        self.asymmetric3_6 = Bottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_p=0.1,
            relu=decoder_relu,
        )
        self.dilated3_7 = Bottleneck(
            128, dilation=16, padding=16, dropout_p=0.1, relu=decoder_relu
        )

        # Stage 4
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_p=0.1, relu=decoder_relu
        )
        self.regular4_1 = Bottleneck(64, padding=1, dropout_p=0.1, relu=decoder_relu)
        self.regular4_2 = Bottleneck(64, padding=1, dropout_p=0.1, relu=decoder_relu)

        # Stage 5
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_p=0.1, relu=decoder_relu
        )
        self.regular5_1 = Bottleneck(16, padding=1, dropout_p=0.1, relu=decoder_relu)

        self.transposed_conv_semantic = nn.ConvTranspose2d(
            16, num_classes, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.transposed_conv_depth = nn.ConvTranspose2d(
            16, 1, kernel_size=3, stride=2, padding=1, bias=False
        )

    def forward(self, x):
        input_size = x.size()
        x = self.initial_block(x)

        ## Encoder ##

        # Stage 1
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        ## Seg Decoder ##
        # Stage 3
        seg = self.regular3_0(x)
        seg = self.dilated3_1(seg)
        seg = self.asymmetric3_2(seg)
        seg = self.dilated3_3(seg)
        seg = self.regular3_4(seg)
        seg = self.dilated3_5(seg)
        seg = self.asymmetric3_6(seg)
        seg = self.dilated3_7(seg)

        # Stage 4
        seg = self.upsample4_0(seg, max_indices2_0, output_size=stage2_input_size)
        seg = self.regular4_1(seg)
        seg = self.regular4_2(seg)

        # Stage 5
        seg = self.upsample5_0(seg, max_indices1_0, output_size=stage1_input_size)
        seg = self.regular5_1(seg)
        seg = self.transposed_conv_semantic(seg, output_size=input_size)

        ## Depth Decoder ##
        # Stage 3
        depth = self.regular3_0(x)
        depth = self.dilated3_1(depth)
        depth = self.asymmetric3_2(depth)
        depth = self.dilated3_3(depth)
        depth = self.regular3_4(depth)
        depth = self.dilated3_5(depth)
        depth = self.asymmetric3_6(depth)
        depth = self.dilated3_7(depth)

        # Stage 4
        depth = self.upsample4_0(depth, max_indices2_0, output_size=stage2_input_size)
        depth = self.regular4_1(depth)
        depth = self.regular4_2(depth)

        # Stage 5
        depth = self.upsample5_0(depth, max_indices1_0, output_size=stage1_input_size)
        depth = self.regular5_1(depth)
        depth = self.transposed_conv_depth(depth, output_size=input_size)

        return seg, depth
