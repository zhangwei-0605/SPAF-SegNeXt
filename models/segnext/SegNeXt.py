import json
import torch.nn as nn
import torch
from segnext import bricks
import torch.nn.functional as F
from timm.models import register_model
from segnext.mscan import MSCAN
from typing import Optional, Union, List
from segmentation_models_pytorch.base.modules import Activation

"""
[batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""


class Hamburger(nn.Module):
    def __init__(
            self,
            hamburger_channels=256,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(Hamburger, self).__init__()
        self.ham_in = bricks.ConvModule(hamburger_channels,
                                        hamburger_channels,
                                        bias=True,
                                        num_groups=0)

        self.ham = bricks.NMF2D(args=nmf2d_config)

        self.ham_out = bricks.ConvModule(hamburger_channels,
                                         hamburger_channels)


    def forward(self, x):
        out = self.ham_in(x)
        out = F.relu(out, inplace=False)
        out = self.ham(out)
        out = self.ham_out(out)
        out = F.relu(x + out, inplace=False)
        return out


class LightHamHead(nn.Module):
    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            num_classes=150,
            drop_prob=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(LightHamHead, self).__init__()

        self.conv_seg = nn.Conv2d(
                in_channels=out_channels,
                out_channels=num_classes,
                kernel_size=(1, 1))

        self.squeeze = bricks.ConvModule(
            in_channels=sum(in_channels_list),
            out_channels=hidden_channels
        )

        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        self.align = bricks.ConvModule(
            in_channels=hidden_channels,
            out_channels=out_channels
        )


    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)

        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        x = torch.cat(inputs, dim=1)

        out = self.squeeze(x)

        out = self.hamburger(out)

        out = self.align(out)

        out = self.conv_seg(out)

        _, _, original_height, original_width = o.shape

        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )

        return out


class SegNeXt(nn.Module):

    def __init__(
            self,
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,

            embed_dims=[32, 64, 160, 256],
            expand_rations=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob_of_encoder=0.1,
            drop_path_prob=0.1,
            hidden_channels=256,
            out_channels=256,
            drop_prob_of_decoder=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cpu"
                }
            ),
            **kwargs
    ):
        super(SegNeXt, self).__init__()

        self.backbone = MSCAN(
            in_chans=3,
            embed_dims=embed_dims,
            mlp_ratios=expand_rations,
            depths=depths,
            drop_rate=drop_prob_of_encoder,
            drop_path_rate=drop_path_prob
        )

        self.decode_head = LightHamHead(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_classes=classes,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

        self.activation = Activation(activation)

    def forward(self, x):
        out = self.backbone(x)
        out = self.decode_head(out)
        out = F.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=True)
        out = self.activation(out)
        return out



def SegNeXt_T(
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            **kwargs):
    embed_dims = [32, 64, 160, 256]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 3, 5, 2]
    hidden_channels = 256
    out_channels = 256

    net = SegNeXt(
                encoder_name =encoder_name,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes=classes,
                activation = activation,
                embed_dims=embed_dims, expand_rations=expand_rations,
                depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                 **kwargs)
    return net


def SegNeXt_S(
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            **kwargs):

    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [2, 2, 4, 2]
    hidden_channels = 256
    out_channels = 256

    net = SegNeXt(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                embed_dims=embed_dims, expand_rations=expand_rations,
                depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                 **kwargs)
    return net


def SegNeXt_B(
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            **kwargs):
    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 3, 12, 3]
    hidden_channels = 512
    out_channels = 512

    net = SegNeXt(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                embed_dims=embed_dims, expand_rations=expand_rations,
                depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                 **kwargs)
    return net


def SegNeXt_L(
            encoder_name: str = "hrnet_w18",
            encoder_weights: Optional[str] = None,
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            **kwargs):
    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 5, 27, 3]
    hidden_channels = 1024
    out_channels = 1024

    net = SegNeXt(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                embed_dims=embed_dims, expand_rations=expand_rations,
                depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                 **kwargs)
    return net




if __name__ == '__main__':
    from torchinfo import summary
    net = SegNeXt_S(classes=6)
    x = torch.randn(2, 3, 512, 512)
    y = net(x)
    print("输出结果：",y.shape)

    summary(net, input_size=(1, 3, 256, 256))
    for name, param in net.named_parameters():
        print(name)


