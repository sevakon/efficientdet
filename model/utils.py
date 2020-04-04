

def efficientdet_params(model_name):
    """ Map EfficientDet model name to parameter coefficients. """
    params_dict = {
        'efficientdet-d0': {'compound_coef': 0, 'backbone': 'efficientnet-b0',
             'R_input': 512, 'W_bifpn': 64, 'D_bifpn': 3, 'D_class': 3, 'params': '3.9M'},

        'efficientdet-d1': {'compound_coef': 1, 'backbone': 'efficientnet-b1',
             'R_input': 640, 'W_bifpn': 88, 'D_bifpn': 4, 'D_class': 3, 'params': '6.6M'},

        'efficientdet-d2': {'compound_coef': 2, 'backbone': 'efficientnet-b2',
             'R_input': 768, 'W_bifpn': 112, 'D_bifpn': 5, 'D_class': 3, 'params': '8.1M'},

        'efficientdet-d3': {'compound_coef': 3, 'backbone': 'efficientnet-b3',
             'R_input': 896, 'W_bifpn': 160, 'D_bifpn': 6, 'D_class': 4, 'params': '12.0M'},

        'efficientdet-d4': {'compound_coef': 4, 'backbone': 'efficientnet-b4',
             'R_input': 1024, 'W_bifpn': 224, 'D_bifpn': 7, 'D_class': 4, 'params': '20.7M'},

        'efficientdet-d5': {'compound_coef': 5, 'backbone': 'efficientnet-b5',
             'R_input': 1280, 'W_bifpn': 288, 'D_bifpn': 7, 'D_class': 4, 'params': '34.3M'},

        'efficientdet-d6': {'compound_coef': 6, 'backbone': 'efficientnet-b6',
             'R_input': 1280, 'W_bifpn': 384, 'D_bifpn': 8, 'D_class': 5, 'params': '51.9M'}
    }
    return params_dict[model_name]


def check_model_name(model_name):
    possibles = ['efficientdet-d' + str(i) for i in range(7)]
    if model_name not in possibles:
        raise ValueError('Name {} not in {}'.format(model_name, possibles))
