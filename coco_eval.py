import os
import time
import json
import torch
import argparse

import config as cfg
from model import EfficientDet
from utils import DetectionWrapper
from pycocotools.cocoeval import COCOeval
from data import create_loader, CocoDetection


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def parse_args():
    parser = argparse.ArgumentParser(description='COCO 2017 validation')

    parser.add_argument('--anno', default='val2017',
                        choices=['train2017', 'val2017', 'test-dev2017'])
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=4, type=int)
    parser.add_argument('--results', default='results.json', type=str)

    args = parser.parse_args()
    return args


def validate(args):
    model = EfficientDet.from_pretrained(cfg.MODEL_NAME)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    wrapper = DetectionWrapper(model.to(device))

    if args.anno == 'test-dev2017':
        raise Exception('test-dev2017 not supported yet')
    else:
        annotation_path = os.path.join(cfg.COCO_PATH, 'annotations', f'instances_{args.anno}.json')
        image_dir = args.anno

    dataset = CocoDetection(os.path.join(cfg.COCO_PATH, image_dir), annotation_path)

    loader = create_loader(
        dataset,
        input_size=cfg.IMAGE_SIZE,
        batch_size=args.batch_size,
        num_workers=args.workers)

    img_ids = []
    results = []
    model.eval()
    batch_time = AverageMeter()
    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            output = wrapper(input, target['img_id'], target['scale'])
            for batch_out in output:
                for det in batch_out:
                    image_id = int(det[0])
                    score = float(det[5])
                    coco_det = {
                        'image_id': image_id,
                        'bbox': det[1:5].tolist(),
                        'score': score,
                        'category_id': int(det[6]),
                    }
                    img_ids.append(image_id)
                    results.append(coco_det)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 1 == 0:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                        .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                    )
                )

    json.dump(results, open(args.results, 'w'), indent=4)
    if 'test' not in args.anno:
        coco_results = dataset.coco.loadRes(args.results)
        coco_eval = COCOeval(dataset.coco, coco_results, 'bbox')
        coco_eval.params.imgIds = img_ids  # score only ids we've used
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return results


if __name__ == '__main__':
    validate(parse_args())
