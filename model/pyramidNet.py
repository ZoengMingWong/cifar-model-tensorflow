# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 15:44:45 2018

@author: hzm
"""

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, is_training, zero_pad=True, name=None):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name=name+'_bn0')
    
    conv1 = layers.conv2d(bn0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'_bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name=name+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn2')
    
    if in_feat.shape != bn2.shape:
        if zero_pad == False:
            short_cut = layers.conv2d(in_feat, out_chans, kernel_size=[1, 1], strides=[stride]*2, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(conv2, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(conv2, in_feat, name=name+'_out_feat')
        
    return out_feat
    
def PreActBottleneck(in_feat, stride, bottleneck, is_training, zero_pad=True, name=None):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name=name+'_bn0')
    
    conv1 = layers.conv2d(bn0, bottleneck, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn2')
    relu2 = tf.nn.relu(bn2, name=name+'_relu2')
    
    conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv3')
    bn3 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn3')
    
    if in_feat.shape != bn3.shape:
        if zero_pad == False:
            short_cut = layers.conv2d(in_feat, bottleneck*4, kernel_size=[1, 1], strides=[stride]*2, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(conv3, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(conv3, in_feat, name=name+'_out_feat')
        
    return out_feat

def PyramidNet(img, alpha, blocks, strides, chans, is_training, zero_pad=True):
    
    stride = []
    units = 0
    for b, s in zip(blocks, strides):
        units += b
        stride += ([s] + [1] * (b - 1))
    add_chans = 1.0 * alpha / units
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    out = layers.batch_normalization(out, is_training, name='img_bn')
    
    for i in range(len(stride)):
        chans += add_chans
        out = PreActBlock(out, stride[i], int(chans), is_training, zero_pad=zero_pad, name='Block_'+str(i))
    
    out = layers.batch_normalization(out, training=is_training, name='end_bn')
    out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def BottleneckResNet(img, alpha, blocks, strides, chans, bottleneck, is_training, zero_pad=True):
    
    stride = []
    units = 0
    for b, s in zip(blocks, strides):
        units += b
        stride += ([s] + [1] * (b - 1))
    add_chans = 1.0 * alpha / units
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    out = layers.batch_normalization(out, is_training, name='img_bn')
    
    for i in range(len(stride)):
        bottleneck += add_chans
        out = PreActBottleneck(out, stride[i], int(bottleneck), is_training, zero_pad=zero_pad, name='Block_'+str(i))
        
    out = layers.batch_normalization(out, training=is_training, name='end_bn')
    out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def PyramidNet_a48_d110(img, is_training, zero_pad=True):
    return PyramidNet(img, 48, [18, 18, 18], [1, 2, 2], 16, is_training=is_training, zero_pad=zero_pad)

def BotPyramidNet_a270_d164(img, is_training, zero_pad=True):
    return BottleneckResNet(img, 270, [18, 18, 18], [1, 2, 2], 16, 16, is_training=is_training, zero_pad=zero_pad)









