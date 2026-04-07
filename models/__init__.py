# -*- encoding: utf-8 -*-

from models.segformer import SegFormer
from models.hrnet import HRNet
from models.unet.unet import Unet
from models.pspnet.pspnet import PSPNet
from models.deeplapv3_plus.deeplabv3_plus import DeepLab
from models.segnext.SegNeXt import SegNeXt_T, SegNeXt_S, SegNeXt_B, SegNeXt_L
from models.segnext_ca.SegNeXt import SegNeXt_T_CA, SegNeXt_S_CA, SegNeXt_B_CA, SegNeXt_L_CA


custom_models = {
    'segformer': SegFormer,
    'hrnet': HRNet,
    'unet': Unet,
    'pspnet': PSPNet,
    'deeplabv3_plus': DeepLab,
    'segnext_t': SegNeXt_T,
    'segnext_s': SegNeXt_S,
    'segnext_b': SegNeXt_B,
    'segnext_l': SegNeXt_L,
    'segnext_t_ca': SegNeXt_T_CA,
    'segnext_s_ca': SegNeXt_S_CA,
    'segnext_b_ca': SegNeXt_B_CA,
    'segnext_l_ca': SegNeXt_L_CA,
}


def create_model(cfg: dict):
    model_type = cfg['type']
    arch = cfg['arch']

    if model_type == 'smp':
        import segmentation_models_pytorch as smp
        smp_net = getattr(smp, arch)

        encoder = cfg.get('encoder', 'resnet34')
        pretrained = cfg.get('pretrained', 'imagenet')
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        aux_params = cfg.get('aux_params', None)

        model = smp_net(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channel,
            classes=out_channel,
            aux_params=aux_params
        )

    elif model_type == 'custom':
        assert arch.lower() in custom_models.keys()
        net = custom_models[arch.lower()]
        encoder = cfg.get('encoder', 'mit_b0')
        pretrained = cfg.get('pretrained', None)
        in_channel = cfg.get('in_channel', 3)
        out_channel = cfg.get('out_channel', 2)
        activation = cfg.get('activation', None)

        model = net(
            encoder_name=encoder,
            encoder_weights=pretrained,
            in_channels=in_channel,
            classes=out_channel,
            activation=activation,
        )

    else:
        print('type error')
        exit()

    return model

if __name__ == '__main__':
    from torchsummary import summary

    from segformer import SegFormer

    encoder = 'mit_b1'
    pretrained = 'imagenet'
    in_channel = 3
    out_channel = 2
    activation = None

    model = SegFormer(
        encoder_name=encoder,
        encoder_weights=pretrained,
        in_channels=in_channel,
        classes=out_channel,
        activation=activation,
    ).cuda()

    summary(model, input_size=(3, 512, 512))
