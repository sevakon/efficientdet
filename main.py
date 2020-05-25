import argparse

import torch
from torch.utils.tensorboard import SummaryWriter

import config as cfg
from dataloader import get_loader
from log.logger import logger
from model import EfficientDet
from train import train
from utils import (CosineLRScheduler, DetectionLoss, ExponentialMovingAverage)
from utils.utils import count_parameters, init_seed
from validation import validate


def parse_args():
    parser = argparse.ArgumentParser(description='Main')

    parser.add_argument('-mode', choices=['trainval', 'eval'],
                        default='trainval', type=str)
    parser.add_argument('-model', default='efficientdet-d0', type=str)
    parser.add_argument('--experiment', type=str, default='experiment')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--cuda', dest='cuda', action='store_true')
    parser.add_argument('--cpu', dest='cuda', action='store_false')
    parser.add_argument('--device', type=int, default=0)
    parser.set_defaults(cuda=True)

    arguments = parser.parse_args()
    return arguments


def build_tools(model):
    optimizer = torch.optim.SGD(
        model.parameters(), lr=cfg.WARMUP_LR,
        weight_decay=cfg.WEIGHT_DECAY, momentum=cfg.MOMENTUM)

    schedule_helper = CosineLRScheduler(
        lr_warmup_init=cfg.WARMUP_LR, base_lr=cfg.BASE_LR,
        lr_warmup_step=cfg.STEPS_PER_EPOCH, total_steps=cfg.TOTAL_STEPS)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: schedule_helper.get_lr_coeff(step))

    criterion = DetectionLoss(
        cfg.ALPHA, cfg.GAMMA, cfg.DELTA, cfg.BOX_LOSS_WEIGHT, cfg.NUM_CLASSES)

    ema_decay = ExponentialMovingAverage(model, cfg.MOVING_AVERAGE_DECAY)

    return optimizer, scheduler, criterion, ema_decay


def setup_writer(tb_tag, args):
    writer = SummaryWriter(log_dir=cfg.LOG_PATH / tb_tag)
    writer.add_text("Hyperparams", '<br />'.join(
        [f"{k}: {v}" for k, v in args.__dict__.items()]))
    writer.train_step, writer.eval_step = 0, 0
    return writer


def main(args):
    device = torch.device('cuda:{}'.format(args.device)) \
        if args.cuda else torch.device('cpu')

    model = EfficientDet.from_pretrained(args.model).to(device) \
        if args.pretrained else EfficientDet.from_name(args.model).to(device)

    if args.mode == 'trainval':
        logger("Model's trainable parameters: {}".format(count_parameters(model)))

        loader = get_loader(path=cfg.TRAIN_SET,
                            annotations=cfg.TRAIN_ANNOTATIONS,
                            batch_size=cfg.BATCH_SIZE)

        optimizer, scheduler, criterion, ema_decay = build_tools(model)
        writer = setup_writer(args.experiment, args)
        best_score = -1

        for epoch in range(cfg.NUM_EPOCHS):
            model, optimizer, scheduler, writer = \
                train(model, optimizer, loader, scheduler,
                      criterion, ema_decay, device, writer)

            if epoch > cfg.VAL_DELAY and \
                    (epoch + 1) % cfg.VAL_INTERVAL == 0:
                ema_decay.assign(model)
                model, writer, best_score = \
                    validate(model, device, writer,
                             cfg.MODEL.SAVE_PATH, best_score=best_score)
                ema_decay.resume(model)

    elif args.mode == 'eval':
        validate(model, device)


if __name__ == '__main__':
    init_seed(cfg.SEED)
    main(parse_args())
