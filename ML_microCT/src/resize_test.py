#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 19:54:06 2018

@author: gtrancourt
"""

# Testing of skimage.transform.resize

from skimage import transform, img_as_int, img_as_ubyte, img_as_float
import numpy as np
import skimage.io as io

def Trim_Individual_Stack(stack, rescale_factor):
    print("***trimming stack***")
    shape_array = np.array(stack.shape) - np.array([np.repeat(0,3), np.repeat(1,3), np.repeat(2,3), np.repeat(3,3), np.repeat(4,3), np.repeat(5,3)])
    dividers_mat = shape_array % rescale_factor
    to_trim = np.argmax(dividers_mat == 0, axis=0)
    for i in np.arange(len(to_trim)):
        if to_trim[i] == 0:
            pass
        else:
            to_delete = np.arange(stack.shape[i]-to_trim[i], stack.shape[i])
            stack = np.delete(stack, to_delete, axis=i)
    return stack


path_to_file = '/Users/gtrancourt/Dropbox/Koelreuteria_paniculata/'
filename = 'Koelreuteria_paniculata-GRID-cropped.tif'

img = io.imread(path_to_file + filename)
img.dtype

rescale_factor = 2

img = Trim_Individual_Stack(img, rescale_factor)
img.size

# With preserve_range
# Still float64, but could be converted to unit8
img_resize_uint8 = transform.resize(img, [img.shape[0],
                                          img.shape[1]/rescale_factor,
                                          img.shape[2]/rescale_factor], 
                                    preserve_range=True)
img_resize_uint8.dtype
img_resize_uint8.size

img_resize_uint8 = np.array(img_resize_uint8.round(), dtype='uint8')
img_resize_uint8.dtype
img_resize_uint8.size

io.imsave(path_to_file + 'test_resize_preserve_range.tif', img_resize_uint8)

img_resize = transform.resize(img, [img.shape[0]/rescale_factor,img.shape[1]/rescale_factor,img.shape[2]/rescale_factor])
img_resize.size



# From http://akuederle.com/create-numpy-array-with-for-loop
result_array = np.empty(np.array(img.shape)/np.array([1,rescale_factor,rescale_factor])) # the default dtype is float, so set dtype if it isn't float

for idx in np.arange(result_array.shape[0]):
    result_array[idx] = transform.resize(img[idx], [img.shape[1]/rescale_factor, img.shape[2]/rescale_factor], preserve_range=True)

%timeit img_resize_uint8 = transform.resize(img, [img.shape[0],img.shape[1]/rescale_factor,img.shape[2]/rescale_factor], preserve_range=True)
%timeit for idx in np.arange(result_array.shape[0]): result_array[idx] = transform.resize(img[idx], [img.shape[1]/rescale_factor, img.shape[2]/rescale_factor], preserve_range=True)


np.sum((img_resize_uint8 - result_array).round(decimals=2), axis=None)

io.imsave(path_to_file + 'test_resize_loop.tif', img_as_ubyte(img_resize_uint8))
