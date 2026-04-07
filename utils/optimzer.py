# https://github.com/open-mmlab/mmcv/tree/master/mmcv/runner/optimizer
import copy
from torch.optim import SGD, RMSprop, Adam, AdamW, Rprop, LBFGS, Adadelta, Adagrad
from typing import Dict, List, Optional, Union
import torch.nn as nn
# 导入必要的模块
import copy
from torch.optim import SGD, RMSprop, Adam, AdamW, Rprop, LBFGS, Adadelta, Adagrad
from typing import Dict, List, Optional, Union
import torch.nn as nn


__all__ = ['build_optimizer']

OPTIMS = [SGD, RMSprop, Adam, AdamW, Rprop, LBFGS, Adadelta, Adagrad]
OPTIMS_dict = {o.__name__.lower(): o for o in OPTIMS}


class OptimizerConstructor:

    def __init__(self, cfg: Dict):
        self.optimizer_type = cfg.pop('type')
        self.paramwise_cfg = cfg.pop('paramwise_cfg', None)
        self.optimizer_cfg = cfg
        self.base_lr = cfg.get('lr', None)
        self.base_wd = cfg.get('weight_decay', None)

    def add_params(self,
                   params: List[Dict],
                   module: nn.Module,
                   prefix: str = ''):
        custom_keys = self.paramwise_cfg.copy() if self.paramwise_cfg else {}
        sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get('bias_lr_mult', 1.) if self.paramwise_cfg else 1.
        bias_decay_mult = self.paramwise_cfg.get('bias_decay_mult', 1.) if self.paramwise_cfg else 1.
        norm_decay_mult = self.paramwise_cfg.get('norm_decay_mult', 1.) if self.paramwise_cfg else 1.
        dwconv_decay_mult = self.paramwise_cfg.get('dwconv_decay_mult', 1.) if self.paramwise_cfg else 1.

        is_norm = isinstance(module, (nn.GroupNorm, nn.LayerNorm))
        is_dwconv = (isinstance(module, nn.Conv2d) and module.in_channels == module.groups)

        for name, param in module.named_parameters(recurse=False):
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            is_custom = False
            for key in sorted_keys:
                if key in f'{prefix}.{name}':
                    is_custom = True
                    lr_mult = custom_keys[key].get('lr_mult', 1.)
                    param_group['lr'] = self.base_lr * lr_mult
                    if self.base_wd is not None:
                        decay_mult = custom_keys[key].get('decay_mult', 1.)
                        param_group['weight_decay'] = self.base_wd * decay_mult
                    break

            if not is_custom:
                if name == 'bias' and not is_norm:
                    param_group['lr'] = self.base_lr * bias_lr_mult
                if self.base_wd is not None:
                    if is_norm:
                        param_group['weight_decay'] = self.base_wd * norm_decay_mult
                    elif is_dwconv:
                        param_group['weight_decay'] = self.base_wd * dwconv_decay_mult
                    elif name == 'bias':
                        param_group['weight_decay'] = self.base_wd * bias_decay_mult

            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix)

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        try:
            optim_class = OPTIMS_dict[self.optimizer_type.lower()]
        except KeyError:
            raise KeyError("错误的优化器类型 `{}`。可用选项为: {}".format(
                self.optimizer_type, list(OPTIMS_dict.keys()),
            ))

        if not self.paramwise_cfg:
            return optim_class(model.parameters(), **self.optimizer_cfg)

        params: List[Dict] = []
        self.add_params(params, model)
        return optim_class(params, **self.optimizer_cfg)


def build_optimizer(model, cfg: Dict):
    optimizer_cfg = copy.deepcopy(cfg)
    optim_constructor = OptimizerConstructor(optimizer_cfg)
    optimizer = optim_constructor(model)
    return optimizer

