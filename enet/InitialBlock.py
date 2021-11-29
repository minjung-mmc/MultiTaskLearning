import torch
import torch.nn as nn
from torch.nn.modules import activation

class InitialBlock(nn.Module):
    def __init__(self, input_ch, output_ch, bias=False, relu=False):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        
        self.main_branch = nn.Conv2d(input_ch, output_ch-3, kernel_size=3, stride=2, padding=1, bias=bias)
        
        self.ext_branch = nn.MaxPool2d(2, stride=2, padding=0)
        
        self.batch_norm = nn.BatchNorm2d(output_ch)

        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)

        out = torch.cat((main,ext),dim=1)

        out = self.batch_norm(out)
        out = self.out_activation(out)

        return out