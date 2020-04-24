import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_


class DetectionLoss(nn.Module):
    """ Overall Detection Loss """

    def __init__(self, alpha, gamma, delta, box_loss_weight):
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.box_loss_weight = box_loss_weight

    def forward(self, cls_outputs, box_outputs, labels):
        cls_losses = []
        box_losses = []

        # for level in

        cls_loss, box_loss = None, None
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss

    def _classification_loss(self, cls_outputs, cls_targets, num_positives):
        pass

    def _regression_loss(self, box_outputs, box_targets, num_positives):
        pass


class ExponentialMovingAverage:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CosineLRScheduler:
    """ Custom Learning Rate Scheduler from paper
    Linear warmup for the first epoch
    Cosine annealing for later epochs
    Args:
        lr_warmup_init (float): initial LR before warmup
        lr_warmup_step (int): finish warmup step
        total_steps (int): total steps
    """

    def __init__(self, lr_warmup_init, base_lr, lr_warmup_step, total_steps):
        self.init_lr = lr_warmup_init
        self.base_lr = base_lr
        self.warmup_steps = lr_warmup_step
        self.total_steps = total_steps

    def get_lr_coeff(self, step):
        """ Returns new LR ratio to previous LR """
        new_lr = self.cosine_lr_schedule(self.base_lr,
            self.init_lr, self.warmup_steps, self.total_steps, step)
        ratio = new_lr / self.init_lr
        return ratio

    @staticmethod
    def cosine_lr_schedule(adjusted_lr, lr_warmup_init, lr_warmup_step,
                           total_steps, step):
        linear_warmup = (lr_warmup_init + step / lr_warmup_step * (
                    adjusted_lr - lr_warmup_init))
        cosine_lr = 0.5 * adjusted_lr * (
                1 + np.cos(np.pi * step / total_steps))
        chosen_lr = linear_warmup if step < lr_warmup_step else cosine_lr
        return chosen_lr


def variance_scaling_(tensor, gain=1.):
    """
    VarianceScaling in https://keras.io/zh/initializers/
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = math.sqrt(gain / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)
