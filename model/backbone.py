import torch
import torch.nn as nn
import torch.nn.functional as F

from model.efficientnet import EfficientNet as EffNet


class EfficientNet(nn.Module):
    """ Backbone Wrapper """
    def __init__(self, model_name):
        super(EfficientNet, self).__init__()
        model = EffNet.from_pretrained(model_name)

            

        print(model.state_dict().keys())
