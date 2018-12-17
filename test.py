# -*- coding: utf-8 -*-

import numpy as np
import os, re
import util

# Notice that the VISIBLE_DEVICES must NOT LESS THAN the gpus used in the checkpoint.
os.environ['CUDA_DEVICES_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    
    data_path = '/home/hzm/cifar_data'
    fs = os.listdir(os.path.join(data_path, 'test'))
    xs_test = np.array([os.path.join(data_path, 'test', f) for f in fs])
    ys_test = np.array([int(re.split('[_.]', f)[1]) for f in fs])
    
    classes = 10
    batch_size = 100
    util.test('ckpt/model-final.meta', 'ckpt/model-final', xs_test, ys_test, batch_size, classes)
