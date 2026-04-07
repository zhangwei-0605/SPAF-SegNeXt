import json
import torch.nn as nn
import torch
from segnext import bricks
import torch.nn.functional as F
from timm.models import register_model
from SegNeXt_BF.mscan import MSCAN
from SegNeXt_BF.FreqFusion import FreqFusion
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


    def _forward_feature(self, inputs):
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

        _, _, original_height, original_width = o.shape

        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )

        return out


class LightHamHeadFreqAware(LightHamHead):

    def __init__(self,
                use_high_pass=True,
                use_low_pass=True,
                compress_ratio=8,
                semi_conv=True,
                low2high_residual=False,
                high2low_residual=False,
                lowpass_kernel=5,
                highpass_kernel=3,
                hamming_window=False,
                feature_resample=True,
                feature_resample_group=4,
                comp_feat_upsample=True,
                use_checkpoint=False,
                feature_resample_norm=True,
                **kwargs):
        super().__init__(**kwargs)
        self.freqfusions = nn.ModuleList()
        in_channels = kwargs.get('in_channels_list', [])

        self.in_channels = in_channels
        self.feature_resample = feature_resample
        self.feature_resample_group = feature_resample_group
        self.use_checkpoint = use_checkpoint
        in_channels = in_channels[::-1]
        pre_c = in_channels[0]
        for c in in_channels[1:]:
            freqfusion = FreqFusion(
                hr_channels=c,
                lr_channels=pre_c,
                scale_factor=1,
                lowpass_kernel=lowpass_kernel,
                highpass_kernel=highpass_kernel,
                up_group=1,
                upsample_mode='nearest',
                align_corners=False,
                feature_resample=feature_resample,
                feature_resample_group=feature_resample_group,
                comp_feat_upsample=comp_feat_upsample,
                hr_residual=True,
                hamming_window=hamming_window,
                compressed_channels= (pre_c + c) // compress_ratio,
                use_high_pass=use_high_pass,
                use_low_pass=use_low_pass,
                semi_conv=semi_conv,
                feature_resample_norm=feature_resample_norm,
                )
            self.freqfusions.append(freqfusion)
            pre_c += c

        assert not (low2high_residual and high2low_residual)
        self.low2high_residual = low2high_residual
        self.high2low_residual = high2low_residual
        if low2high_residual:
            self.low2high_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.low2high_convs.append(nn.Conv2d(pre_c, c, 1))
                pre_c = c
        elif high2low_residual:
            self.high2low_convs = nn.ModuleList()
            pre_c = in_channels[0]
            for c in in_channels[1:]:
                self.high2low_convs.append(nn.Conv2d(c, pre_c, 1))
                pre_c += c

    def _forward_feature(self, inputs):
        inputs = inputs[::-1]
        in_channels = self.in_channels[::-1]
        lowres_feat = inputs[0]
        if self.low2high_residual:
            for pre_c, hires_feat, freqfusion, low2high_conv in zip(in_channels[:-1], inputs[1:], self.freqfusions, self.low2high_convs):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                lowres_feat = torch.cat([hires_feat + low2high_conv(lowres_feat[:, :pre_c]), lowres_feat], dim=1)
            pass
        else:
            for idx, (hires_feat, freqfusion) in enumerate(zip(inputs[1:], self.freqfusions)):
                _, hires_feat, lowres_feat = freqfusion(hr_feat=hires_feat, lr_feat=lowres_feat, use_checkpoint=self.use_checkpoint)
                if self.feature_resample:
                    b, _, h, w = hires_feat.shape
                    lowres_feat = torch.cat([hires_feat.reshape(b * self.feature_resample_group, -1, h, w),
                                             lowres_feat.reshape(b * self.feature_resample_group, -1, h, w)], dim=1).reshape(b, -1, h, w)
                else:
                    lowres_feat = torch.cat([hires_feat, lowres_feat], dim=1)

        inputs = lowres_feat
        x = self.squeeze(inputs)
        x = self.hamburger(x)
        output = self.align(x)

        return output

    def forward(self, inputs):
        output = self._forward_feature(inputs)
        output = self.conv_seg(output)
        return output



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
                    "device": "cuda"
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

        self.decode_head = LightHamHeadFreqAware(
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


def SegNeXt_TF(
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


def SegNeXt_SF(
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


def SegNeXt_BF(
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


def SegNeXt_LF(
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


