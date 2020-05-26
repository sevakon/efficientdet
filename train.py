import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import config as cfg
from utils import DetectionTrainWrapper
from utils.utils import get_gradnorm, get_lr, is_valid_number


def train(model, optimizer, loader, scheduler, criterion, ema, device, writer):
    model.train()

    progress_bar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for step, batch in progress_bar:
        x, labels = batch['img'], batch['annotation']
        gt_labels, gt_boxes = labels[:, :, 4], labels[:, :, :4]
        batch_size = x.shape[0]

        wrapper = DetectionTrainWrapper(model, device, criterion)
        loss, cls_loss, box_loss = wrapper(x, gt_labels, gt_boxes)

        values = [v.data.item() for v in [loss, cls_loss, box_loss]]

        progress_bar.set_description(
            "all:{0:.2f} | cls:{1:.2f} | box:{2:.2f}".format(
                values[0], values[1], values[2]))

        if is_valid_number(loss.data.item()):
            loss.backward()

            writer.add_scalar('Train/overall_loss', values[0], writer.train_step)
            writer.add_scalar('Train/class_loss', values[1], writer.train_step)
            writer.add_scalar('Train/box_loss', values[2], writer.train_step)
            writer.add_scalar('Train/gradnorm', get_gradnorm(optimizer), writer.train_step)
            writer.add_scalar('Train/lr', get_lr(optimizer), writer.train_step)
            writer.add_scalar(f"Train/gpu memory", torch.cuda.memory_allocated(device), writer.train_step)

            writer.train_step += 1
            writer.flush()

            clip_grad_norm_(model.parameters(), cfg.CLIP_GRADIENTS_NORM)
            optimizer.step()
            optimizer.zero_grad()

            ema(model, step // batch_size)

            scheduler.step()

    return model, optimizer, scheduler, writer
