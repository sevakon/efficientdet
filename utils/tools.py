import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_


class DetectionLoss(nn.Module):
    """ Overall Detection Loss """

    def __init__(self, alpha, gamma, delta, box_loss_weight, levels=5):
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.box_loss_weight = box_loss_weight
        self.levels = levels

    def forward(self, cls_outputs, box_outputs,
                cls_targets, box_targets, num_positives):
        cls_losses, box_losses = [], []

        for level in range(self.levels):
            cls_loss = self._classification_loss(
                cls_outputs[level], cls_targets[level], num_positives)
            box_loss = self._regression_loss(
                box_outputs[level], box_targets[level], num_positives)
            cls_losses.append(cls_loss)
            box_losses.append(box_loss)

        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss

    def _classification_loss(self, cls_outputs, cls_targets, num_positives):
        normalizer = num_positives
        classification_loss = focal_loss(cls_outputs, cls_targets,
                                         self.alpha, self.gamma, normalizer)
        return classification_loss

    def _regression_loss(self, box_outputs, box_targets, num_positives):
        normalizer = num_positives * 4.0
        mask = box_targets != 0.0
        box_loss = huber_loss(box_targets, box_outputs, weights=mask,
                              delta=self.delta, size_average=False)
        box_loss /= normalizer
        return box_loss


def huber_loss(input, target, delta=1., weights=None, size_average=True):
    err = input - target
    abs_err = err.abs()
    quadratic = torch.clamp(abs_err, max=delta)
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if weights is not None:
        loss *= weights
    return loss.mean() if size_average else loss.sum()


def focal_loss(logits, targets, alpha, gamma, normalizer):
    """Compute the focal loss between `logits` and the golden `target` values.
    Focal loss = -(1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    Args:
      logits: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      targets: A float32 tensor of size
        [batch, height_in, width_in, num_predictions].
      alpha: A float32 scalar multiplying alpha to the loss from positive examples
        and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      normalizer: A float32 scalar normalizes the total loss from all examples.
    Returns:
      loss: A float32 scalar representing normalized total loss.
    """

    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits - gamma * torch.log1p(torch.exp(neg_logits)))
    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    weighted_loss /= normalizer
    return weighted_loss


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
    var_scaled = _no_grad_normal_(tensor, 0., std)
    return var_scaled
