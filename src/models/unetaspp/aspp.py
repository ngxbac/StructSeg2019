import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    def __init__(self, num_channels):
        super(GroupNorm32, self).__init__(num_channels=num_channels, num_groups=32)

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, norm_layer):
        super(_ASPPModule, self).__init__()
        self.norm = norm_layer(planes)
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.elu = nn.ELU(True)

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.norm(x)
        return self.elu(x)

class ASPP(nn.Module):
    def __init__(self, dilations, inplanes, planes, norm_layer, dropout=0.5):
        super(ASPP, self).__init__()

        self.aspp1 = _ASPPModule(inplanes, planes, 1, padding=0, dilation=dilations[0], norm_layer=norm_layer)
        self.aspp2 = _ASPPModule(inplanes, planes, 3, padding=dilations[1], dilation=dilations[1], norm_layer=norm_layer)
        self.aspp3 = _ASPPModule(inplanes, planes, 3, padding=dilations[2], dilation=dilations[2], norm_layer=norm_layer)
        self.aspp4 = _ASPPModule(inplanes, planes, 3, padding=dilations[3], dilation=dilations[3], norm_layer=norm_layer)

        self.norm1 = norm_layer(planes)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
                                             norm_layer(planes),
                                             nn.ELU(True))
        self.conv1 = nn.Conv2d(5 * planes, planes, 1, bias=False)
        self.elu = nn.ELU(True)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.elu(x)

        return self.dropout(x)
