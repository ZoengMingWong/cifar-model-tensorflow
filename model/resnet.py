# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:59:46 2018

@author: hzm
"""

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, is_training, zero_pad=False, name=None):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name=name+'_bn0')
    relu0 = tf.nn.relu(bn0, name=name+'_relu0')
    
    conv1 = layers.conv2d(relu0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'_bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name=name+'_conv2')
    
    if in_feat.shape != conv2.shape:
        if zero_pad == False:
            short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(conv2, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(conv2, in_feat, name=name+'_out_feat')
        
    return out_feat
    
    
def BasicBlock(in_feat, stride, out_chans, is_training, zero_pad=False, name=None):
    
    conv1 = layers.conv2d(in_feat, out_chans, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'_bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name=name+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn2')
    relu2 = tf.nn.relu(bn2, name=name+'_relu2')
    
    if in_feat.shape != relu2.shape:
        if zero_pad == False:
            short_cut = layers.conv_shortcut(in_feat, out_chans, [stride]*2, training=is_training, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(relu2, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(relu2, in_feat, name=name+'_out_feat')
        
    return out_feat
    
def PreActBottleneck(in_feat, stride, bottleneck, is_training, zero_pad=False, name=None):
    
    bn0 = layers.batch_normalization(in_feat, training=is_training, name=name+'_bn0')
    relu0 = tf.nn.relu(bn0, name=name+'_relu0')
    
    conv1 = layers.conv2d(relu0, bottleneck, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn2')
    relu2 = tf.nn.relu(bn2, name=name+'_relu2')
    
    conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv3')
    
    if in_feat.shape != conv3.shape:
        if zero_pad == False:
            short_cut = layers.conv2d(relu0, bottleneck*4, kernel_size=[1, 1], strides=[stride]*2, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(conv3, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(conv3, in_feat, name=name+'_out_feat')
        
    return out_feat
    
def Bottleneck(in_feat, stride, bottleneck, is_training, zero_pad=False, name=None):
    
    conv1 = layers.conv2d(in_feat, bottleneck, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv1')
    bn1 = layers.batch_normalization(conv1, training=is_training, name=name+'_bn1')
    relu1 = tf.nn.relu(bn1, name=name+'_relu1')
    
    conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name=name+'_conv2')
    bn2 = layers.batch_normalization(conv2, training=is_training, name=name+'_bn2')
    relu2 = tf.nn.relu(bn2, name=name+'_reku2')
    
    conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name=name+'_conv3')
    bn3 = layers.batch_normalization(conv3, training=is_training, name=name+'_bn3')
    relu3 = tf.nn.relu(bn3, name=name+'_relu3')
    
    if in_feat.shape != relu3.shape:
        if zero_pad == False:
            short_cut = layers.conv_shortcut(in_feat, bottleneck*4, [stride]*2, training=is_training, name=name+'_short_cut')
        else:
            short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name=name+'_short_cut')
        out_feat = tf.add(relu3, short_cut, name=name+'_out_feat')
    else:
        out_feat = tf.add(relu3, in_feat, name=name+'_out_feat')
        
    return out_feat
    
def ResNet(img, blocks, strides, chans, is_training, preAct=True, zero_pad=False):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    if preAct == False:
        out = layers.batch_normalization(out, is_training, name='img_bn')
        out = tf.nn.relu(out, name='img_relu')
    
    for i in range(len(stride)):
        chans *= stride[i]
        if preAct == True:
            out = PreActBlock(out, stride[i], chans, is_training, zero_pad=zero_pad, name='Block_'+str(i))
        else:
            out = BasicBlock(out, stride[i], chans, is_training, zero_pad=zero_pad, name='Block_'+str(i))
    if preAct == True:
        out = layers.batch_normalization(out, training=is_training, name='end_bn')
        out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def BottleneckResNet(img, blocks, strides, chans, bottleneck, is_training, preAct=True, zero_pad=False):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='img_conv')
    if preAct == False:
        out = layers.batch_normalization(out, is_training, name='img_bn')
        out = tf.nn.relu(out, name='img_relu')
    
    for i in range(len(stride)):
        bottleneck *= stride[i]
        if preAct == True:
            out = PreActBottleneck(out, stride[i], bottleneck, is_training, zero_pad=zero_pad, name='Block_'+str(i))
        else:
            out = Bottleneck(out, stride[i], bottleneck, is_training, zero_pad=zero_pad, name='Block_'+str(i))
    if preAct == True:
        out = layers.batch_normalization(out, training=is_training, name='end_bn')
        out = tf.nn.relu(out, name='end_relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out

def ResNet18(img, is_training, zero_pad=False):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=False, zero_pad=zero_pad)
    
def PreResNet18(img, is_training, zero_pad=False):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=True, zero_pad=zero_pad)
    
def ResNet50(img, is_training, zero_pad=False):
    return BottleneckResNet(img, [3, 4, 6, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=False, zero_pad=zero_pad)
    
def PreResNet50(img, is_training, zero_pad=False):
    return BottleneckResNet(img, [3, 4, 6, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=True, zero_pad=zero_pad)




