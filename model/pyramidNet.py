# -*- coding: utf-8 -*-

import tensorflow as tf
from . import layers

def PreActBlock(in_feat, stride, out_chans, is_training, zero_pad=True, name=None):
    with tf.variable_scope(name, default_name='preBlock'):
        bn0 = layers.batch_normalization(in_feat, training=is_training, name='bn0')
        
        conv1 = layers.conv2d(bn0, out_chans, kernel_size=[3, 3], strides=[stride]*2, name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, out_chans, kernel_size=[3, 3], strides=[1, 1], name='conv2')
        bn2 = layers.batch_normalization(conv2, training=is_training, name='bn2')
        
        if in_feat.shape[1:] != bn2.shape[1:]:
            if zero_pad == False:
                short_cut = tf.nn.relu(bn0, name='relu0')
                short_cut = layers.conv2d(short_cut, out_chans, [1, 1], [stride]*2, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, out_chans, [stride]*2, name='zero_shortcut')
            out_feat = tf.add(bn2, short_cut, name='add')
        else:
            out_feat = tf.add(bn2, in_feat, name='add')
    return out_feat
    
def PreActBottleneck(in_feat, stride, bottleneck, is_training, zero_pad=True, name=None):
    with tf.variable_scope(name, default_name='preBotBlock'):
        bn0 = layers.batch_normalization(in_feat, training=is_training, name='bn0')
        
        conv1 = layers.conv2d(bn0, bottleneck, kernel_size=[1, 1], strides=[1, 1], name='conv1')
        bn1 = layers.batch_normalization(conv1, training=is_training, name='bn1')
        relu1 = tf.nn.relu(bn1, name='relu1')
        
        conv2 = layers.conv2d(relu1, bottleneck, kernel_size=[3, 3], strides=[stride]*2, name='conv2')
        bn2 = layers.batch_normalization(conv2, training=is_training, name='bn2')
        relu2 = tf.nn.relu(bn2, name='relu2')
        
        conv3 = layers.conv2d(relu2, bottleneck*4, kernel_size=[1, 1], strides=[1, 1], name='conv3')
        bn3 = layers.batch_normalization(conv3, training=is_training, name='bn3')
        
        if in_feat.shape[1:] != bn3.shape[1:]:
            if zero_pad == False:
                short_cut = tf.nn.relu(bn0, name='relu0')
                short_cut = layers.conv2d(short_cut, bottleneck*4, [1, 1], [stride]*2, name='conv_shortcut')
            else:
                short_cut = layers.zero_pad_shortcut(in_feat, bottleneck*4, [stride]*2, name='pad_shortcut')
            out_feat = tf.add(bn3, short_cut, name='add')
        else:
            out_feat = tf.add(bn3, in_feat, name='add')
    return out_feat

def PyramidNet(img, alpha, blocks, strides, chans, is_training, zero_pad=True):
    
    stride = []
    units = 0
    for b, s in zip(blocks, strides):
        units += b
        stride += ([s] + [1] * (b - 1))
    add_chans = 1.0 * alpha / units
    
    with tf.variable_scope('Begin'):
        out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='iconv')
        out = layers.batch_normalization(out, is_training, name='bn')
    
    for i in range(len(stride)):
        chans += add_chans
        out = PreActBlock(out, stride[i], int(chans), is_training, zero_pad=zero_pad, name='Block_'+str(i))
        
    with tf.variable_scope('End'):
        out = layers.batch_normalization(out, training=is_training, name='bn')
        out = tf.nn.relu(out, name='relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def BotPyramidNet(img, alpha, blocks, strides, chans, bottleneck, is_training, zero_pad=True):
    
    stride = []
    units = 0
    for b, s in zip(blocks, strides):
        units += b
        stride += ([s] + [1] * (b - 1))
    add_chans = 1.0 * alpha / units
    
    with tf.variable_scope('Begin'):
        out = layers.conv2d(img, chans, kernel_size=[3, 3], strides=[1, 1], name='conv')
        out = layers.batch_normalization(out, is_training, name='bn')
    
    for i in range(len(stride)):
        bottleneck += add_chans
        out = PreActBottleneck(out, stride[i], int(bottleneck), is_training, zero_pad=zero_pad, name='Block_'+str(i))
    
    with tf.variable_scope('End'):
        out = layers.batch_normalization(out, training=is_training, name='bn')
        out = tf.nn.relu(out, name='relu')
    out = tf.reduce_mean(out, axis=[1, 2], name='global_avg_pooling')
    return out
    
def PyramidNet_a48_d110(img, is_training, zero_pad=True):
    return PyramidNet(img, 48, [18, 18, 18], [1, 2, 2], 16, is_training=is_training, zero_pad=zero_pad)

def BotPyramidNet_a270_d164(img, is_training, zero_pad=True):
    return BotPyramidNet(img, 270, [18, 18, 18], [1, 2, 2], 16, 16, is_training=is_training, zero_pad=zero_pad)
    
def BotPyramidNet_a200_d272(img, is_training, zero_pad=True):
    return BotPyramidNet(img, 200, [30, 30, 30], [1, 2, 2], 16, 16, is_training=is_training, zero_pad=zero_pad)










