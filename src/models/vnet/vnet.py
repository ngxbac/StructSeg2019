import torch
import torch.nn as nn
import torch.nn.functional as F


from segmentation_models_pytorch.base.model import Model
from segmentation_models_pytorch.base.encoder_decoder import EncoderDecoder
# from segmentation_models_pytorch.common.blocks import Conv2dReLU
from segmentation_models_pytorch.encoders import get_encoder

from .cbam import CBAM_Module
from .aspp import ASPP, GroupNorm32


class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x


class ConvBnRelu2d(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, kernel_size=(3,3), stride=(1,1), padding=(1,1)):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = norm_layer(out_channels)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 norm_layer=GroupNorm32,
                 reduction=16,
                 attention_kernel_size=3,
                 attention_type='none',
                 reslink=False,
        ):
        super().__init__()
        self.block = nn.Sequential(
            ConvBnRelu2d(in_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer),
            ConvBnRelu2d(out_channels, out_channels, kernel_size=3, padding=1, norm_layer=norm_layer),
        )

        self.attention = not attention_type == 'none'
        self.attention_type = attention_type
        self.reslink = reslink

        if self.attention_type.find('cbam') >= 0:
            self.channel_gate = CBAM_Module(out_channels, reduction, attention_kernel_size)

        if self.reslink:
            self.shortcut = ConvBn2d(
                in_channels,
                out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                norm_layer=norm_layer
            )

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if skip is not None:
            x = torch.cat([x, skip], dim=1)

        if self.reslink:
            shortcut = self.shortcut(x)
        x = self.block(x)
        if self.attention:
            x = self.channel_gate(x)
        if self.reslink:
            x = F.relu(x + shortcut)
        return x


class CenterBlock(DecoderBlock):

    def forward(self, x):
        return self.block(x)


class UnetDecoder(Model):

    def __init__(
            self,
            encoder_channels,
            decoder_channels=(256, 128, 64, 32, 16),
            final_channels=1,
            center='none',
            group_norm=False,
            reslink=False,
            attention_type='none',
            multi_task=False
    ):
        super().__init__()

        norm_layer = GroupNorm32 if group_norm else nn.BatchNorm2d

        if center.find('none') >= 0:
            self.center = None
        else:
            channels = encoder_channels[0]
            if center.find('normal') >= 0:
                self.center = CenterBlock(
                    channels, channels,
                    norm_layer=norm_layer,
                    attention_type=attention_type,
                    reslink=reslink
                )
            elif center.find('aspp') >= 0:
                self.center = ASPP(channels, channels, norm_layer)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(
            in_channels[0],
            out_channels[0],
            norm_layer=norm_layer,
            attention_type=attention_type,
            reslink=reslink
        )
        self.layer2 = DecoderBlock(
            in_channels[1],
            out_channels[1],
            norm_layer=norm_layer,
            attention_type=attention_type,
            reslink=reslink
        )
        self.layer3 = DecoderBlock(
            in_channels[2],
            out_channels[2],
            norm_layer=norm_layer,
            attention_type=attention_type,
            reslink=reslink
        )
        self.layer4 = DecoderBlock(
            in_channels[3],
            out_channels[3],
            norm_layer=norm_layer,
            attention_type=attention_type,
            reslink=reslink
        )
        self.layer5 = DecoderBlock(
            in_channels[4],
            out_channels[4],
            norm_layer=norm_layer,
            attention_type=attention_type,
            reslink=reslink
        )
        self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1, 1))

        self.multi_task = multi_task
        if self.multi_task:
            self.fc_1 = nn.Linear(encoder_channels[0], final_channels)

        self.initialize()

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2],
            0 + decoder_channels[3],
        ]
        return channels

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        if self.multi_task:
            encoder_features = F.adaptive_avg_pool2d(encoder_head, 1)
            encoder_features = encoder_features.view(encoder_features.size(0), -1)
            # print(encoder_features.shape)
            x_logit = self.fc_1(encoder_features)

        if self.center:
            encoder_head = self.center(encoder_head)

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])
        x = self.layer4([x, skips[3]])
        x = self.layer5([x, None])
        x = self.final_conv(x)

        if self.multi_task:
            return x, x_logit
        else:
            return x


class VNet(EncoderDecoder):
    def __init__(
            self,
            encoder_name='resnet34',
            encoder_weights='imagenet',
            group_norm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=1,
            activation='sigmoid',
            center='none',  # usefull for VGG models
            attention_type=None,
            reslink=False,
            multi_task=False
    ):
        assert center in ['none', 'normal', 'aspp']
        assert attention_type in ['none', 'cbam']

        print("**" * 50)
        print("Encoder name: \t\t{}".format(encoder_name))
        print("Center: \t\t{}".format(center))
        print("Attention type: \t\t{}".format(attention_type))
        print("Reslink: \t\t{}".format(reslink))

        encoder = get_encoder(
            encoder_name,
            encoder_weights=encoder_weights
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            group_norm=group_norm,
            center=center,
            attention_type=attention_type,
            reslink=reslink,
            multi_task=multi_task
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'vnet-{}'.format(encoder_name)
