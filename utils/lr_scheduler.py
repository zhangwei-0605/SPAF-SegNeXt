# https://blog.csdn.net/qq_31580989/article/details/121491181
# https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html
import warnings
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):

    def __init__(self,
                 optimizer,
                 power=1.0,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 min_lr=0,
                 last_epoch=-1,
                 verbose=False):

        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} 不是 Optimizer 类型'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.epochs = epochs
        self.min_lr = min_lr
        self.power = power

        param_dic = {'total_steps': total_steps, 'epochs': epochs, 'steps_per_epoch': steps_per_epoch}
        for k, v in param_dic.items():
            if v is not None:
                if v <= 0 or not isinstance(v, int):
                    raise ValueError("期望 {} 为正整数，但实际得到 {}".format(k, v))

        if total_steps is not None:
            self.total_steps = total_steps
        elif epochs is not None and steps_per_epoch is None:
            self.total_steps = epochs
        elif epochs is not None and steps_per_epoch is not None:
            self.total_steps = epochs * steps_per_epoch
        else:
            raise ValueError("必须定义 total_steps 或 epochs 或 (epochs 和 steps_per_epoch)")

        super(PolyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def _format_param(self, name, optimizer, param):
        if isinstance(param, (list, tuple)):
            if len(param) != len(optimizer.param_groups):
                raise ValueError(
                    "期望 {} 的值数量为 {}, 实际得到 {}".format(name, len(optimizer.param_groups), len(param)))
            return param
        else:
            return [param] * len(optimizer.param_groups)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("要获取调度器计算的最后一个学习率，请使用 `get_last_lr()`.", UserWarning)

        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("尝试执行 {} 次步进操作。指定的总步数为 {}".format(step_num + 1, self.total_steps))

        coeff = (1 - step_num / self.total_steps) ** self.power

        return [(base_lr - self.min_lr) * coeff + self.min_lr for base_lr in self.base_lrs]

