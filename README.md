# EfficientDet: Scalable and Efficient Object Detection

### PyTorch Implementation of the state-of-the-art object detection architecture EfficientDet 
https://arxiv.org/abs/1911.09070

<img src="https://sun9-35.userapi.com/c205628/v205628726/d29b4/gTjpU4gj2zc.jpg">


### Motivation
As of the time I started working on this project, there was no PyTorch implementation on GitHub that would match the original paper in the number of the model's parameters.

### Model Zoo
| Model Name | Weights | #params | #params paper | mAP | mAP paper |
| :----------: | :--------: | :-----------: | :--------: | :-----: | :-----: |
| D0 | coming soon | 3.878M | 3.9M | soon | 33.5 | 
| D1 | coming soon | 6.622M | 6.6M | soon | 39.1 |
| D2 | coming soon | 8.091M | 8.1M | soon | 42.5 |
| D3 | coming soon | 12.022M | 12.0M | soon | 45.9 |
| D4 | coming soon | 20.708M | 20.7M | soon | 49.0 |
| D5 | coming soon | 33.633M | 33.7M | soon | 50.5 |


### RoadMap
- [X] Model Architecture that would match the original paper
- [X] COCO train and val script 
- [X] port weights from TensorFlow 
- [ ] Reproduce results from the paper
- [ ] Pre-trained weights release

### References
- EfficientDet: Scalable and Efficient Object Detection [arXiv:1911.09070](https://arxiv.org/abs/1911.09070)
- EfficientDet implementation in TensorFlow by [Google AutoML](https://github.com/google/automl/tree/master/efficientdet)
- PyTorch EfficientNet implementation by [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)
