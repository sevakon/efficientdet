import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

import config as cfg
from utils.utils import get_gradnorm, get_lr, is_valid_number


def train(model, optimizer, loader, scheduler, criterion, ema, device, writer):
    model.train()

    pbar = tqdm(enumerate(loader), total=len(loader), leave=False)
    for step, batch in pbar:

        batch_size = batch.shape[0]
        x, labels = batch
        cls_output, box_output = model(x)

        loss, cls_loss, box_loss = criterion(cls_output, box_output, labels)
        values = [v.data.item() for v in [loss, cls_loss, box_loss]]

        pbar.set_description(
            "all:{.2f} | cls:{.2f} | box:{.2f}".format(
                values[0], values[1], values[2])
        )

        if is_valid_number(loss.data.item()):
            loss.backward()

            writer.add_scalar('Train/overall_loss', values[0], writer.train_step)
            writer.add_scalar('Train/class_loss', values[1], writer.train_step)
            writer.add_scalar('Train/box_loss', values[2], writer.train_step)
            writer.add_scalar('Train/gradnorm', get_gradnorm(optimizer), writer.train_step)
            writer.add_scalar('Train/lr', get_lr(optimizer), writer.train_step)
            writer.add_scalar(f"Train/gpu memory", torch.cuda.memory_allocated(device), writer.train_step)

            writer.train_step += 1

            clip_grad_norm_(model.parameters(), cfg.CLIP_GRADIENTS_NORM)
            optimizer.step()
            optimizer.zero_grad()

            ema(model, step // batch_size)

            scheduler.step()

    return model, optimizer, scheduler, writer
