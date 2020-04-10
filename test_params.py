import torch
from model import EfficientDet
import config as cfg
from model.utils import efficientdet_params, count_parameters


""" Quick test on parameters number """


model = EfficientDet.from_pretrained().to('cpu')

model.train()
params = count_parameters(model)

print('Model: {}, params: {:.6f}M, params in paper: {}'.format(cfg.MODEL_NAME, params / 1e6,
                                                     efficientdet_params(cfg.MODEL_NAME)['params']))
print('   Backbone: {:.6f}M'.format(count_parameters(model.backbone) / 1e6))
print('   Adjuster: {:.6f}M'.format(count_parameters(model.adjuster) / 1e6))
print('      BiFPN: {:.6f}M'.format(count_parameters(model.bifpn) / 1e6))
print('       Head: {:.6f}M'.format((count_parameters(model.classifier) +
                                    count_parameters(model.regresser)) / 1e6))

# model.initialize_weights()

image_size = efficientdet_params(cfg.MODEL_NAME)['R_input']
x = torch.rand(1, 3, image_size, image_size)
box, cls = model(x)

for b, c in zip(box, cls):
    print(b.shape)
    print(c.shape)
