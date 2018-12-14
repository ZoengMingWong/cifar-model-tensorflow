# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 15:59:46 2018

@author: hzm
"""

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, is_training, zero_pad=False, name=None):
    with tf.variable_scope(name, default_name='preBlock'):
        bn0 = layers.batch_normalization(in_feat, training=is_training, name='bn0')
        relu0 = tf.nn.relu(bn0, name='relu0')
        
        conv1 = layers.conv2d(relu0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='conv2')
        
        if in_feat.shape[1:] != conv2.shape[1:]:
            if zero_pad == False:
                short_cut = layers.conv2d(relu0, out_chans, kernel_size=[1, 1], strides=[stride]*2, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(conv2, short_cut, name='add')
        else:
            out_feat = tf.add(conv2, in_feat, name='add')
    return out_feat
    
def BasicBlock(in_feat, stride, out_chans, is_training, zero_pad=False, name=None):
    with tf.variable_scope(name, default_name='BasicBlock'):
        conv1 = layers.conv2d(in_feat, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='conv2')
        bn2 = layers.batch_normalization(conv2, training=is_training, name='bn2')
        
        if in_feat.shape[1:] != bn2.shape[1:]:
            if zero_pad == False:
                short_cut = layers.conv_shortcut(in_feat, out_chans, [stride]*2, training=is_training, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(bn2, short_cut, name='add')
        else:
            out_feat = tf.add(bn2, in_feat, name='add')
        out_feat = tf.nn.relu(out_feat, name='add_relu')
    return out_feat
    
def PreActBottleneck(in_feat, stride, bottleneck, is_training, zero_pad=False, name=None):
    with tf.variable_scope(name, default_name='preBotBlock'):
        bn0 = layers.batch_normalization(in_feat, training=is_training, name='bn0')
        relu0 = tf.nn.relu(bn0, name='relu0')
        
        conv1 = layers.conv2d(relu0, bottleneck, kernel_size=[1, 1], strides=[1, 1], name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name='conv2')
        bn2 = layers.batch_normalization(conv2, training=is_training, name='bn2')
        relu2 = tf.nn.relu(bn2, name='relu2')
        
        conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name='conv3')
        
        if in_feat.shape[1:] != conv3.shape[1:]:
            if zero_pad == False:
                short_cut = layers.conv2d(relu0, bottleneck*4, kernel_size=[1, 1], strides=[stride]*2, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(conv3, short_cut, name='add')
        else:
            out_feat = tf.add(conv3, in_feat, name='add')
    return out_feat
    
def Bottleneck(in_feat, stride, bottleneck, is_training, zero_pad=False, name=None):
    with tf.variable_scope(name, default_name='BotBlock'):
        conv1 = layers.conv2d(in_feat, bottleneck, kernel_size=[1, 1], strides=[1, 1], name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name='conv2')
        bn2 = layers.batch_normalization(conv2, training=is_training, name='bn2')
        relu2 = tf.nn.relu(bn2, name='relu2')
        
        conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name='conv3')
        bn3 = layers.batch_normalization(conv3, training=is_training, name='bn3')
        
        if in_feat.shape[1:] != bn3.shape[1:]:
            if zero_pad == False:
                short_cut = layers.conv_shortcut(in_feat, bottleneck*4, [stride]*2, training=is_training, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(bn3, short_cut, name='add')
        else:
            out_feat = tf.add(bn3, in_feat, name='add')
        out_feat = tf.nn.relu(out_feat, name='add_relu')
    return out_feat
    
def ResNet(img, blocks, strides, chans, is_training, preAct=True, zero_pad=False):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    with tf.variable_scope('Begin'):
        out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='conv')
        if preAct == False:
            out = layers.batch_normalization(out, is_training, name='bn')
            out = tf.nn.relu(out, name='relu')
    
    for i in range(len(stride)):
        chans *= stride[i]
        if preAct == True:
            out = PreActBlock(out, stride[i], chans, is_training, zero_pad=zero_pad, name='Block_'+str(i))
        else:
            out = BasicBlock(out, stride[i], chans, is_training, zero_pad=zero_pad, name='Block_'+str(i))
    if preAct == True:
        with tf.variable_scope('End'):
            out = layers.batch_normalization(out, training=is_training, name='bn')
            out = tf.nn.relu(out, name='relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def BottleneckResNet(img, blocks, strides, chans, bottleneck, is_training, preAct=True, zero_pad=False):
    
    stride = []
    for b, s in zip(blocks, strides):
        stride += ([s] + [1] * (b - 1))
    
    with tf.variable_scope('Begin'):
        out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='conv')
        if preAct == False:
            out = layers.batch_normalization(out, is_training, name='bn')
            out = tf.nn.relu(out, name='relu')
    
    for i in range(len(stride)):
        bottleneck *= stride[i]
        if preAct == True:
            out = PreActBottleneck(out, stride[i], bottleneck, is_training, zero_pad=zero_pad, name='Block_'+str(i))
        else:
            out = Bottleneck(out, stride[i], bottleneck, is_training, zero_pad=zero_pad, name='Block_'+str(i))
    if preAct == True:
        with tf.variable_scope('End'):
            out = layers.batch_normalization(out, training=is_training, name='bn')
            out = tf.nn.relu(out, name='relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out

def ResNet18(img, is_training, zero_pad=False):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=False, zero_pad=zero_pad)
    
def PreResNet18(img, is_training, zero_pad=False):
    return ResNet(img, [2, 2, 2, 2], [1, 2, 2, 2], 64, is_training=is_training, preAct=True, zero_pad=zero_pad)
    
def PreResNet50(img, is_training, zero_pad=False):
    return BottleneckResNet(img, [3, 4, 6, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=True, zero_pad=zero_pad)
    
def PreResNet101(img, is_training, zero_pad=False):
    return BottleneckResNet(img, [3, 4, 23, 3], [1, 2, 2, 2], 64, 64, is_training=is_training, preAct=True, zero_pad=zero_pad)



