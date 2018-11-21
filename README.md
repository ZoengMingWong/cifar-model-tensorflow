# Cifar10
The Tensorflow implementation of some models like ResNet and WideResNet on the Cifar-10 or Cifar-100 dataset. There are also some novel methods such as data augmentation in this project.
## Dataset
Download the python version [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) or [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) dataset from http://www.cs.toronto.edu/~kriz/cifar.html and unzip the dataset.  
For convenience, the project doesn't use the original dataset as most other projects do, however, the dataset will be firstly convert to PNG images that every image contains its label in the filename, e.g. the image "test0_3.png" represents the 1st image of the testing dataset with which the label is 3. It is memory efficient if we can dynamically parse the images while needed. However, it doesn't waste too much time as we can use the multiprocessing to parallelly prepare the datasets for the next epoch while training.  
You can do it with the command below. Note that the project doesn't provide the command line parameters, you should config them in the corresponding files.  
`python cifar_to_png.py`
## Reference
### ResNet
> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [_Deep Residual Learning for Image Recognition_](https://arxiv.org/abs/1512.03385). arXiv:1512.03385v1 [cs.CV] 10 Dec 2015.  

> Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. [_Identity Mappings in Deep Residual Networks_](https://arxiv.org/abs/1603.05027). arXiv:1603.05027v3 [cs.CV] 25 Jul 2016.  
### ResNeXt
> Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. [_Aggregated Residual Transformations for Deep Neural Networks_](https://arxiv.org/abs/1611.05431). arXiv:1611.05431v2 [cs.CV] 11 Apr 2017.
### WideResNet
> Sergey Zagoruyko, Nikos Komodakis. [_Wide Residual Networks_](https://arxiv.org/abs/1605.07146v4). arXiv:1605.07146v4 [cs.CV] 14 Jun 2017.  

Pytorch implementation:  https://github.com/szagoruyko/wide-residual-networks
### PyramidNet
> Dongyoon Han, Jiwhan Kim, Junmo Kim. [_Deep Pyramidal Residual Networks_](https://arxiv.org/abs/1610.02915v4). arXiv:1610.02915v4 [cs.CV] 6 Sep 2017.  

Torch implementation: https://github.com/jhkim89/PyramidNet
### Mixup Augmentation
> Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz. [_Mixup: Beyond Empirical Risk Minimization_](https://arxiv.org/abs/1710.09412). arXiv:1710.09412v2 [cs.LG] 27 Apr 2018.  

Pytorch implementation: https://github.com/facebookresearch/mixup-cifar10
### Cutout Augmentation
> Terrance DeVries, Graham W. Taylor. [_Improved Regularization of Convolutional Neural Networks with Cutout_](https://arxiv.org/abs/1708.04552v2). arXiv:1708.04552v2 [cs.CV] 29 Nov 2017.
### AutoAugmentation
> Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le. [_AutoAugment: Learning Augmentation Policies from Data_](https://arxiv.org/abs/1805.09501v2). arXiv:1805.09501v2 [cs.CV] 9 Oct 2018.  

Tensorflow implementation: https://github.com/tensorflow/models/tree/master/research/autoaugment
