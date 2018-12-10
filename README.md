# Cifar10
The Tensorflow implementation of some models including the ResNet, ResNeXt, WideResNet, PyramidNet, etc. on the Cifar-10 or Cifar-100 dataset. There are also some novel augmentation and regularization methods such as Mixup and autoAugmentation and ShakeDrop in this project. The Code is clear to read and understand, and you can learn some useful Python and TensorFlow skills including multiprocessing, pyplot, PIL, and how to feed the network with tf.data.Dataset, how to accelerate the training with multiGPUs, how to make a checkpoint and restore it, and so on.  

Under construction ...
## Usage
### Dataset
Download the python version [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) or [CIFAR-100](http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz) dataset from http://www.cs.toronto.edu/~kriz/cifar.html and unzip the dataset.

For convenience, the project doesn't use the original dataset as most other projects do, however, the dataset will be firstly converted to PNG images that every image contains its label in the filename, e.g. the image "test0_3.png" represents the 1st image of the testing dataset with which the label is 3. It is memory efficient if we can dynamically parse the images while needed, e.g. using the map function in tf.data.Dataset. However, it doesn't waste too much time as we can use the multiprocessing to parallelly prepare the datasets for the next epoch while training.

You can do it with the command below in the linux shell. Note that the project doesn't provide the command line parameters, you should config them in the corresponding files. You can check whether the images is correctly saved after the command.

`$ python cifar_to_png.py`
### Training
As mentioned above, the command line parameters are not provided, but you can easily config the parameters in the begining of the codes. There are two versions of training files, the single/no GPU one and the multiGPUs one. The single/no GPU one, named `train.py`, can be executed without any GPUs but just CPUs (although it's very slow), or with only one GPU. On the contrary, the multiGPUs one, named `train_multigpus.py`, must executed with at least one GPU, with data parallelism, which means that all GPUs have the same compute graph. Both two codes would saved the best result as a checkpoint while training, and you can test or retrain the model with a checkpoint by running the `test.py` or `train_with_ckpt.py` and its multiGPUs version `train_with_ckpt_multigpus.py`, respectively.  

For simplicity, run the command line bewlow in the shell, Windows DOS supported as well, and *tensorflow*, *numpy*, *PIL*, *matplotlib*, *multiprocessing*, *cPickle*, *re* etc. are needed. By default the *preact ResNet18* model with *Mixup* augmentation will be trained.

`$ python train.py`
## Result
Here are some results on Cifar-10.\n
The result of the model preact ResNet18 with autoAugmentation method. The test error rate is **4.37%** for the last epoch.  

`lr = 0.1 if epoch <= 100 else lr = 0.01 if epoch <= 150 else lr = 0.001,  
epochs = 200, weight_decay = 1e-4, gradient_clipping = 5.0, momentum = 0.9, use_nestrov = True`  

![image](https://github.com/ZoengMingWong/cifar-model-tensorflow/blob/master/preresnet18_autoAug.png)
The result of the model WideResNet-28-10 with autoAugmentation. The test error rate is **2.79%** for the last epoch.  

`lr is cosine learning rate like autoAugmentation,  
epochs = 200, weight_decay = 5e-4, gradient_clipping = 5.0, momentum = 0.9, use_nestrov = True`  

![image](https://github.com/ZoengMingWong/cifar-model-tensorflow/blob/master/wrn28_autoAug.png)
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
### ShakeDrop
> Yoshihiro Yamada, Masakazu Iwamura, Koichi Kise. [_ShakeDrop regularization_](https://arxiv.org/abs/1802.02375v1). arXiv:1802.02375v1 [cs.CV] 7 Feb 2018.  
### Mixup Augmentation
> Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz. [_Mixup: Beyond Empirical Risk Minimization_](https://arxiv.org/abs/1710.09412). arXiv:1710.09412v2 [cs.LG] 27 Apr 2018.  

Pytorch implementation: https://github.com/facebookresearch/mixup-cifar10
### Cutout Augmentation
> Terrance DeVries, Graham W. Taylor. [_Improved Regularization of Convolutional Neural Networks with Cutout_](https://arxiv.org/abs/1708.04552v2). arXiv:1708.04552v2 [cs.CV] 29 Nov 2017.
### AutoAugmentation
> Ekin D. Cubuk, Barret Zoph, Dandelion Mane, Vijay Vasudevan, Quoc V. Le. [_AutoAugment: Learning Augmentation Policies from Data_](https://arxiv.org/abs/1805.09501v2). arXiv:1805.09501v2 [cs.CV] 9 Oct 2018.  

Tensorflow implementation: https://github.com/tensorflow/models/tree/master/research/autoaugment
