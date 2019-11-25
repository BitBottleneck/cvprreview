# Bit Bottleneck

### Introduction
This is a novel bitwise compressing method used to quantize and compress the activation of deep neural networks.

### Dependencies

+ Python 3. 6+
+ Tensorflow 1. 4. 0+
+ Sklearn 1. 10. 0+
+ Pycharm 2018. 3. 3

### Usage

+ Download the CIFAR10 dataset.
+ Set `rounding=True` in `resnet.py` to gain the pre-training parameters.
+ Set `print_feature=True` in `resnet.py` to print the activations of network model.
+ Run `calculate_optimal_alpha.py`, to calculate the sparse coefficients of alpha, where you can set 
   the threshold of PSNR loss.
+ Set `rounding=False` and `print_feature=False`, which meas to use Bit Bottleneck to compress the activation, 
    and set the `is_use_ckpt` to False in `hyper_parameters.py` to gain the initial parameters of new model whose 
    Bit Bottleneck layers are initialized by vector alpha in olny one iteration.
+ Run `transfer_learning.py`, to transmit the pre-training parameters to coresponding layers except Bit Bottleneck layer.
+ Set the `is_use_ckpt` to True, to retrain the new model in small amount iteration.

Note the change in parameter address when making transfer learning, and the hyper parameters setting which can be found in the paper.

### Result

The result can be found in the paper.

### Acknowledgment

This project referes to  (https://github.com/wenxinxu/resnet_in_tensorflow)
