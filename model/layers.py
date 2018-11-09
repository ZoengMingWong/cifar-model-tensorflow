# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 16:23:26 2018

@author: hzm
"""
import tensorflow as tf

def conv2d(in_feat, out_chans, kernel_size, strides, padding='SAME', activation=None, use_bias=False, weight_decay=0., name=None):
    shape = kernel_size + [in_feat.shape[-1].value, out_chans]
    strides = [1] + strides + [1]
    elements = 1.0
    for i in shape:
        elements *= i
    stdv = 1.0 / tf.sqrt(elements)
    weight = tf.Variable(tf.random_uniform(shape, -stdv, stdv), name=(None if name is None else name+'_weight'))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(weight))
    
    conv = tf.nn.conv2d(in_feat, weight, strides, padding, name=(None if name is None else name+'_conv'))
    
    if use_bias:
        bias = tf.Variable(tf.random_uniform([out_chans], -stdv, stdv), name=(None if name is None else name+'_bias'))
        conv = tf.add(conv, bias, name=(None if name is None else name+'_add_bias'))
    if activation is not None:
        conv = activation(conv, name=(None if name is None else name+'_activation'))
    return conv
    
def linear(in_feat, out_classes, activation=False, use_bias=False, weight_decay=0., name=None):
    shape = [in_feat.shape[-1].value, out_classes]
    elements = 1.0
    for i in shape:
        elements *= i
    stdv = 1.0 / tf.sqrt(elements)
    weight = tf.Variable(tf.random_uniform(shape, -stdv, stdv), name=(None if name is None else name+'_weight'))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(weight_decay)(weight))
    
    full = tf.matmul(in_feat, weight, name=(None if name is None else name+'_full_connect'))
    
    if use_bias:
        bias = tf.Variable(tf.random_uniform([out_classes], -stdv, stdv), name=(None if name is None else name+'_bias'))
        full = tf.add(full, bias, name=(None if name is None else name+'_add_bias'))
    if activation is not None:
        full = activation(full, name=(None if name is None else name+'_activation'))
    return full

def batch_normalization(in_feat, training, name):
    out_feat = tf.layers.batch_normalization(in_feat, epsilon=1e-5, momentum=0.9, training=training, name=name)
    return out_feat
