# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 17:12:49 2018

@author: hzm
"""

from __future__ import division
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

def translate(img, x=0., y=0.):
    size = img.size
    affine_tuple = (1, 0, size[0]*x, 0, 1, size[1]*y)
    new_img = img.transform(img.size, Image.AFFINE, affine_tuple)
    return new_img
    
def translateX(img, x=0.):
    return translate(img, x=x)
    
def translateY(img, y=0.):
    return translate(img, y=y)
    
def shear(img, x=0, y=0):
    affine_tuple = (1, x, 0, y, 1, 0)
    new_img = img.transform(img.size, Image.AFFINE, affine_tuple)
    return new_img
    
def shearX(img, x=0):
    return shear(img, x=x)
    
def shearY(img, y=0):
    return shear(img, y=y)
    
def rotate(img, degree=0, center=True):
    theta = degree * np.pi / 180.0
    cos_ = np.cos(theta)
    sin_ = np.sin(theta)
    size = img.size
    x, y = size[0]/2, size[1]/2
    if not center:
        affine_tuple = (cos_, -sin_, 0, sin_, cos_, 0)
        new_img = img.transform(size, Image.AFFINE, affine_tuple)
    else:
        affine_tuple = (cos_, -sin_, x-x*cos_+y*sin_, sin_, cos_, y-x*sin_-y*cos_)
        new_img = img.transform(size, Image.AFFINE, affine_tuple)
    return new_img
    
def cutout(img, size=16):
    img_size = img.size
    pad = ImageOps.expand(img, tuple([size]*4))
    
    mask = Image.new('RGB', (size, size), color=0)
    rnd_x = np.random.randint(0, size+img_size[0])
    rnd_y = np.random.randint(0, size+img_size[1])
    
    pad.paste(mask, (rnd_x, rnd_y))
    new_img = ImageOps.crop(pad, tuple([size]*4))
    return new_img
    
def autoContrast(img, paras=None):
    new_img = ImageOps.autocontrast(img)
    return new_img
    
def equalize(img, paras=None):
    new_img = ImageOps.equalize(img)
    return new_img

def invert(img, paras=None):
    new_img = ImageOps.invert(img)
    return new_img
    
def solarize(img, thres=256):
    new_img = ImageOps.solarize(img, thres)
    return new_img
    
def posterize(img, bits=0):
    new_img = ImageOps.posterize(img, bits)
    return new_img
    
def contrast(img, factor=1.0):
    new_img = ImageEnhance.Contrast(img).enhance(factor)
    return new_img
    
def sharpness(img, factor=1.0):
    new_img = ImageEnhance.Sharpness(img).enhance(factor)
    return new_img
    
def color(img, factor=1.0):
    new_img = ImageEnhance.Color(img).enhance(factor)
    return new_img
    
def brightness(img, factor=1.0):
    new_img = ImageEnhance.Brightness(img).enhance(factor)
    return new_img
    
def pad_to_bounding_box(img, left, upper, target_width, target_height):
    img_size = img.size
    right = target_width - img_size[0] - left
    lower = target_height - img_size[1] - upper
    new_img = ImageOps.expand(img, (left, upper, right, lower))
    return new_img
    
def random_crop(img, size=[32, 32]):
    img_size = img.size
    rnd_x = np.random.randint(0, img_size[0]-size[0])
    rnd_y = np.random.randint(0, img_size[1]-size[1])
    new_img = ImageOps.crop(img, (rnd_x, rnd_y, img_size[0]-size[0]-rnd_x, img_size[1]-size[1]-rnd_y))
    return new_img
    
def random_flip_left_right(img):
    rnd = np.random.uniform()
    if rnd < 0.5:
        img = ImageOps.mirror(img)
    return img

def per_image_standarization(img):
    img = np.array(img).astype('float') / 255.0
    elements = img.size
    std_img = (img - np.mean(img)) / np.max([np.std(img), 1.0/np.sqrt(elements)])
    return std_img




    
    