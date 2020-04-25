import config as cfg
from model import EfficientDet
from utils.utils import count_parameters

""" Quick test on parameters number """

model = EfficientDet.from_pretrained('efficientdet-d0').to('cpu')

model.eval()

print('Model: {}, params: {:.6f}M, params in paper: {}'.format(
    cfg.MODEL_NAME, count_parameters(model) / 1e6, cfg.PARAMS))
print('   Backbone: {:.6f}M'.format(
    count_parameters(model.backbone) / 1e6))
print('   Adjuster: {:.6f}M'.format(
    count_parameters(model.adjuster) / 1e6))
print('      BiFPN: {:.6f}M'.format(
    count_parameters(model.bifpn) / 1e6))
print('       Head: {:.6f}M'.format(
   (count_parameters(model.classifier) +
    count_parameters(model.regresser)) / 1e6))
