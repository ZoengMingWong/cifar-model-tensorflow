# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:25:26 2018

@author: hzm
"""

from __future__ import division
import ImageOperation as ImOps
import numpy as np
            
ops = {'ShearX': ImOps.shearX,          'ShearY': ImOps.shearY, 
       'TranslateX': ImOps.translateX,  'TranslateY': ImOps.translateY, 
       'Rotate': ImOps.rotate,          'Solarize': ImOps.solarize, 
       'Posterize': ImOps.posterize,    'Contrast': ImOps.contrast,
       'Color': ImOps.color,            'Brightness': ImOps.brightness, 
       'Sharpness': ImOps.sharpness,    'Cutout': ImOps.cutout, 
       'AutoContrast': ImOps.autoContrast, 
       'Invert': ImOps.invert,          'Equalize': ImOps.equalize}
           
rangeMag = {'ShearX': [-0.3, 0.3],      'ShearY': [-0.3, 0.3], 
            'TranslateX': [-10, 10],  'TranslateY': [-10, 10], 
            'Rotate': [-30, 30],        'Solarize': [0, 256], 
            'Posterize': [4, 8],        'Contrast': [0.1, 1.9],
            'Color': [0.1, 1.9],        'Brightness': [0.1, 1.9], 
            'Sharpness': [0.1, 1.9],    'Cutout': [0, 20], 
            'AutoContrast': [0, 0],     'Invert': [0, 0],   'Equalize': [0, 0]}
            
intMag = { 'ShearX': False,         'ShearY': False, 
           'TranslateX': True,      'TranslateY': True, 
           'Rotate': False,         'Solarize': True, 
           'Posterize': True,       'Contrast': False,
           'Color': False,          'Brightness': False, 
           'Sharpness': False,      'Cutout': True, 
           'AutoContrast': False, 
           'Invert': False,         'Equalize': False}

def D2A(rangeA, digit, toInt=False):
    analog = digit * (rangeA[1] - rangeA[0]) / 10.0 + rangeA[0]
    if toInt == True:
        analog = int(analog)
    return analog

def ImOperation(img, ImOp, prob, mag):
    mag = D2A(rangeMag[ImOp], mag, intMag[ImOp])
    rnd = np.random.uniform()
    if rnd <= prob and prob > 0.0:
        img = ops[ImOp](img, mag)
    return img
    
def policy(img):
    policy_id = np.random.randint(0, 25)
    img = eval('policy'+str(policy_id)+'(img)')
    return img

def policy0(img):
    img = ImOperation(img, 'Invert', 0.1, 7)
    img = ImOperation(img, 'Contrast', 0.2, 6)
    return img
    
def policy1(img):
    img = ImOperation(img, 'Rotate', 0.7, 2)
    img = ImOperation(img, 'TranslateX', 0.3, 9)
    return img

def policy2(img):
    img = ImOperation(img, 'Sharpness', 0.8, 1)
    img = ImOperation(img, 'Sharpness', 0.9, 3)
    return img
    
def policy3(img):
    img = ImOperation(img, 'ShearY', 0.5, 8)
    img = ImOperation(img, 'TranslateY', 0.7, 9)
    return img
    
def policy4(img):
    img = ImOperation(img, 'AutoContrast', 0.5, 8)
    img = ImOperation(img, 'Equalize', 0.9, 2)
    return img
    
def policy5(img):
    img = ImOperation(img, 'ShearY', 0.2, 7)
    img = ImOperation(img, 'Posterize', 0.3, 7)
    return img
    
def policy6(img):
    img = ImOperation(img, 'Color', 0.4, 3)
    img = ImOperation(img, 'Brightness', 0.6, 7)
    return img
    
def policy7(img):
    img = ImOperation(img, 'Sharpness', 0.3, 9)
    img = ImOperation(img, 'Brightness', 0.7, 9)
    return img
    
def policy8(img):
    img = ImOperation(img, 'Equalize', 0.6, 5)
    img = ImOperation(img, 'Equalize', 0.5, 1)
    return img
    
def policy9(img):
    img = ImOperation(img, 'Contrast', 0.6, 7)
    img = ImOperation(img, 'Sharpness', 0.6, 5)
    return img

def policy10(img):
    img = ImOperation(img, 'Color', 0.7, 7)
    img = ImOperation(img, 'TranslateX', 0.5, 8)
    return img
    
def policy11(img):
    img = ImOperation(img, 'Equalize', 0.3, 7)
    img = ImOperation(img, 'AutoContrast', 0.4, 8)
    return img

def policy12(img):
    img = ImOperation(img, 'TranslateY', 0.4, 3)
    img = ImOperation(img, 'Sharpness', 0.2, 6)
    return img
    
def policy13(img):
    img = ImOperation(img, 'Brightness', 0.9, 6)
    img = ImOperation(img, 'Color', 0.2, 8)
    return img
    
def policy14(img):
    img = ImOperation(img, 'Solarize', 0.5, 2)
    img = ImOperation(img, 'Invert', 0.0, 3)
    return img
    
def policy15(img):
    img = ImOperation(img, 'Equalize', 0.2, 0)
    img = ImOperation(img, 'AutoContrast', 0.6, 0)
    return img
    
def policy16(img):
    img = ImOperation(img, 'Equalize', 0.2, 8)
    img = ImOperation(img, 'Equalize', 0.6, 4)
    return img
    
def policy17(img):
    img = ImOperation(img, 'Color', 0.9, 9)
    img = ImOperation(img, 'Equalize', 0.6, 6)
    return img
    
def policy18(img):
    img = ImOperation(img, 'AutoContrast', 0.8, 4)
    img = ImOperation(img, 'Solarize', 0.2, 8)
    return img
    
def policy19(img):
    img = ImOperation(img, 'Brightness', 0.1, 3)
    img = ImOperation(img, 'Color', 0.7, 0)
    return img
    
def policy20(img):
    img = ImOperation(img, 'Solarize', 0.4, 5)
    img = ImOperation(img, 'AutoContrast', 0.9, 3)
    return img
    
def policy21(img):
    img = ImOperation(img, 'TranslateY', 0.9, 9)
    img = ImOperation(img, 'TranslateY', 0.7, 9)
    return img

def policy22(img):
    img = ImOperation(img, 'AutoContrast', 0.9, 2)
    img = ImOperation(img, 'Solarize', 0.8, 3)
    return img
    
def policy23(img):
    img = ImOperation(img, 'Equalize', 0.8, 8)
    img = ImOperation(img, 'Invert', 0.1, 3)
    return img
    
def policy24(img):
    img = ImOperation(img, 'TranslateY', 0.7, 9)
    img = ImOperation(img, 'AutoContrast', 0.9, 1)
    return img

    
    
    
    
    
    
