# Bit Bottleneck
### Introduction
This a novel bit-wise compression method used to compress the activation of deep neural networks.

### Dependencies
+ Python 3.6+
+ TensorFlow >= 1.4.0+
+ sklearn 1.10.0

### Usage

  +Download the CIFAR10 dataset.
  +Set rounding = True to gain the pre-train parameter.
  +Set print_feature = Ture to print the activation.
  +Run calculate_optimal_alpha.py to gain the sparse coefficents of alpha, where you could set the threshold of PSNR loss.
  +Set rounding = False and print_feature = False, which means use Bit Bottleneck layer, and set is_use_ckpt to False 
    to gain a initial train parameter with only one step.
  +Run transfer_learning.py to transmit the parameter to corresponding layers except Bit Bottleneck layers, then set is_use_ckpt to 
    True to retrain the new model inserted with Bit Bottleneck layers.

Note the change in the file path, and the hyper parameters setting which you can find in the paper. 
