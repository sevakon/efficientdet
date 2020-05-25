# EfficientDet: Scalable and Efficient Object Detection

### PyTorch Implementation of the state-of-the-art object detection architecture EfficientDet 
https://arxiv.org/abs/1911.09070

<img src="https://sun9-35.userapi.com/c205628/v205628726/d29b4/gTjpU4gj2zc.jpg">


### Motivation
As of the time I started working on this project, there was no PyTorch implementation on GitHub that
 would match the original paper in the number of the model's parameters. 
All of the existed repositories altered a lot from the recently published TensorFlow 
implementation by [Brain Team](https://github.com/google/automl/tree/master/efficientdet) (e.g. changing strides in the backbone,
missing batch normalization layers, no 'same' padding strategy in pooling layers, differing training hyper-parameters, not using Exponential Moving Average Decay, and others). 
Here is my attempt to reproduce EfficientDet in PyTorch.
My end goal is to reproduce training cycle from the original paper and achieve nearly same results.

### Notes on Implementation
Alternatively to the TensorFlow implementation, I got rid of the useless biases
in convolutional layers followed by batch normalization, which resulted in 
**parameters reduction**.

### Model Zoo
| Model Name | Weights | #params | #params paper | val mAP | val mAP paper |
| :----------: | :--------: | :-----------: | :--------: | :-----: | :-----: |
| D0 | [download](https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d0.pth) | 3.878M | 3.9M | 32.8 | 33.5 | 
| D1 | [download](https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d1.pth) | 6.622M | 6.6M | 38.7 | 39.1 |
| D2 | [download](https://github.com/sevakon/efficientdet/releases/download/2.0/efficientdet-d2.pth) | 8.091M | 8.1M | 42.1 | 42.5 |
| D3 | soon | 12.022M | 12.0M | soon | 45.9 |
| D4 | soon | 20.708M | 20.7M | soon | 49.0 |
| D5 | soon | 33.633M | 33.7M | soon | 50.5 |

### Usage

#### Train from scratch

##### Download COCO2017 Train & Val Sets
```bash
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && mv train2017 data/coco/train2017 && rm train2017.zip

wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && mv val2017 data/coco/val2017 && rm val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && mv annotations data/coco && rm annotations_trainval2017.zip
```

##### Run Script

```bash
python main.py -mode 'trainval' -model 'efficientdet-d{}'
```

#### COCO Evaluation

##### Download COCO2017 Val Set
```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip && mv val2017 data/coco/val2017 && rm val2017.zip

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && mv annotations data/coco && rm annotations_trainval2017.zip
```

##### Run Script

```bash
python main.py -mode eval -model efficientdet-d{} --pretrained
```


### RoadMap
- [X] Model Architecture that would match the original paper
- [X] COCO val script 
- [X] port weights from TensorFlow 
- [X] COCO train script
- [ ] Reproduce results from the paper
- [ ] Pre-trained weights release

### References
- EfficientDet: Scalable and Efficient Object Detection [arXiv:1911.09070](https://arxiv.org/abs/1911.09070)
- EfficientDet implementation in TensorFlow by [Google AutoML](https://github.com/google/automl/tree/master/efficientdet)
- PyTorch EfficientNet implementation by [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)
