import torch
import torch.nn as nn
import torch.nn.functional as F

from model.efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):
    """ Backbone Wrapper """
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(model_name)
        del model._fc
        self.model = model

    def forward(self, x):
        self.model.forward(x)

        # x = self.model._swish(self.model._bn0(self.model._conv_stem(x)))
        #
        # features = []
        # for idx, block in enumerate(self.model._blocks):
        #     drop_connect_rate = self.model._global_params.drop_connect_rate
        #     if drop_connect_rate:
        #         drop_connect_rate *= float(idx) / len(self.model._blocks)
        #     x = block(x, drop_connect_rate=drop_connect_rate)
        #     # if block._depthwise_conv.stride == [2, 2]:
        #     features.append(x)
        #
        # x = self.model._swish(self.model._bn1(self.model._conv_head(x)))
        # features.append(x)

        # 40 64 64
        # 80 32 32
        # 112 16 16
        # 192 8 8
        # 320 4 4

        return features[1:]



if __name__ == '__main__':
    ''' quick test '''

    backbone = EfficientNet('')
