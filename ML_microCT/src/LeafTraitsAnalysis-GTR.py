#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 20:03:05 2018

@author: gtrancourt
"""

#%%
import cStringIO
import numpy as np
import os
from pandas import DataFrame
from PIL import Image
from skimage import transform, img_as_bool, img_as_int, img_as_ubyte, img_as_float32
import skimage.io as io
from skimage.measure import label, marching_cubes_lewiner, mesh_surface_area, regionprops
import zipfile
#import cv2

def Trim_Individual_Stack(large_stack, small_stack):
    print("***trimming stack***")
    dims = np.array(binary_stack.shape, dtype='float') / np.array(raw_pred_stack.shape, dtype='float')
    if np.all(dims <= 2):
        return large_stack
    else:
        if dims[1] > 2:
            if (large_stack.shape[1]-1)/2 == small_stack.shape[1]:
                large_stack = np.delete(large_stack, large_stack.shape[1]-1, axis=1)
            else:
                if (large_stack.shape[1]-2)/2 == small_stack.shape[1]:
                    large_stack = np.delete(large_stack, np.arange(large_stack.shape[1]-2, large_stack.shape[1]), axis=1)
        if dims[2] > 2:
            if (large_stack.shape[2]-1)/2 == small_stack.shape[2]:
                large_stack = np.delete(large_stack, large_stack.shape[2]-1, axis=2)
            else:
                if (large_stack.shape[2]-2)/2 == small_stack.shape[2]:
                    large_stack = np.delete(large_stack, np.arange(large_stack.shape[2]-2, large_stack.shape[2]), axis=2)
        return large_stack
    

#%%
#Pixel dimmension
px_edge = 0.1625 #µm
vx_volume = px_edge**3

#Define de values of the different tissues
epid_value = 69
bg_value = 177
spongy_value = 0
palisade_value = 0
if spongy_value == palisade_value:
    mesophyll_value = spongy_value
else:
    mesophyll_value = [spongy_value, palisade_value]
ias_value = 255
vein_value = 147

#%%

#Load segmented image
base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/_ML_DONE/'
sample_name = 'C_D_10_Strip1_'
folder_name = 'MLresults/'
binary_filename = sample_name + 'BINARY-8bit.tif'
raw_ML_prediction_name = 'fullstack_prediction.tif'

filepath = base_folder_name + sample_name + '/'

#%%
if os.path.isfile(filepath + sample_name + 'RESULTS.txt'):
    print('This file has already been processed!')
    assert False

    
#%% 
# Pre-processing on the raw ML predictions
raw_pred_stack = io.imread(filepath + folder_name + raw_ML_prediction_name)
print(np.unique(raw_pred_stack[100]))
io.imshow(raw_pred_stack[100])

#%%
# Sometimes, the output values are different than what is written above.
# This is a way to assign values. But you have to double-check that the other is right.

# mesophyll_value, epid_value, unknown_value_1, unknown_value_2, vein_value, bg_value, ias_value = np.split(np.unique(raw_pred_stack[100]), len(np.unique(raw_pred_stack[100])))


#%%
###################
## EPIDERMIS
###################

# Get the epidermis regions and remove the unconnected epidermis
unique_epidermis_volumes = label(raw_pred_stack == epid_value, connectivity=1)
props_of_unique_epidermis = regionprops(unique_epidermis_volumes)

io.imshow(unique_epidermis_volumes[100])
#%%
epidermis_area = np.zeros(len(props_of_unique_epidermis))
epidermis_label = np.zeros(len(props_of_unique_epidermis))
epidermis_centroid = np.zeros([len(props_of_unique_epidermis),3])
for regions in np.arange(len(props_of_unique_epidermis)):
    epidermis_area[regions] = props_of_unique_epidermis[regions].area
    epidermis_label[regions] = props_of_unique_epidermis[regions].label
    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

# Find the largest epidermis
ordered_epidermis = np.argsort(epidermis_area)
print(epidermis_area[ordered_epidermis[-10:]])
print(epidermis_centroid[ordered_epidermis[-4:]])

two_largest_epidermis = (unique_epidermis_volumes == ordered_epidermis[-1]+1) | (unique_epidermis_volumes == ordered_epidermis[-2]+1)

#Check if it's correct
#io.imsave(filepath + folder_name + 'test_epidermis.tif', 
#          img_as_ubyte(two_largest_epidermis))
io.imshow(two_largest_epidermis[100])
#%%

# Get the values again
unique_epidermis_volumes = label(two_largest_epidermis, connectivity=1)
props_of_unique_epidermis = regionprops(unique_epidermis_volumes)
epidermis_area = np.zeros(len(props_of_unique_epidermis))
epidermis_label = np.zeros(len(props_of_unique_epidermis))
epidermis_centroid = np.zeros([len(props_of_unique_epidermis),3])
for regions in np.arange(len(props_of_unique_epidermis)):
    epidermis_area[regions] = props_of_unique_epidermis[regions].area
    epidermis_label[regions] = props_of_unique_epidermis[regions].label
    epidermis_centroid[regions] = props_of_unique_epidermis[regions].centroid

#io.imshow(unique_epidermis_volumes[100])

unique_epidermis_volumes = np.array(unique_epidermis_volumes, dtype='uint8')

adaxial_epidermis_value = unique_epidermis_volumes[100,:,100][(unique_epidermis_volumes[100,:,100] != 0).argmax()]
abaxial_epidermis_value = int(np.arange(start=1,stop=3)[np.arange(start=1,stop=3) != adaxial_epidermis_value])

epidermis_adaxial_volume = epidermis_area[adaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)
epidermis_abaxial_volume = epidermis_area[abaxial_epidermis_value - 1] * (px_edge * (px_edge*2)**2)

epidermis_abaxial_thickness = np.sum((unique_epidermis_volumes == abaxial_epidermis_value), axis=1) * (px_edge*2)
epidermis_adaxial_thickness = np.sum((unique_epidermis_volumes == adaxial_epidermis_value), axis=1) * (px_edge*2)

#%%
###################
## VEINS
###################

# Get the veins volumes
unique_vein_volumes = label(raw_pred_stack == vein_value, connectivity=1)
props_of_unique_veins = regionprops(unique_vein_volumes)

io.imshow(unique_vein_volumes[100])
#%%
veins_area = np.zeros(len(props_of_unique_veins))
veins_label = np.zeros(len(props_of_unique_veins))
veins_centroid = np.zeros([len(props_of_unique_veins),3])
for regions in np.arange(len(props_of_unique_veins)):
    veins_area[regions] = props_of_unique_veins[regions].area
    veins_label[regions] = props_of_unique_veins[regions].label
    veins_centroid[regions] = props_of_unique_veins[regions].centroid

# Find the largest veins
ordered_veins = np.argsort(veins_area)
veins_area[ordered_veins[-80:]]
veins_area[ordered_veins[:1000]]
veins_centroid[ordered_veins[-4:]]

np.sum(veins_area <= 1000)

large_veins_ids = veins_label[veins_area > 100000]

largest_veins = np.in1d(unique_vein_volumes, large_veins_ids).reshape(raw_pred_stack.shape)

# Get the values again
vein_volume = np.sum(largest_veins) * (px_edge * (px_edge*2)**2)

#Check if it's correct
#io.imsave(base_folder_name + sample_name + '/' + folder_name + 'test_veins.tif', 
#          img_as_ubyte(largest_veins))
io.imshow(largest_veins[100])


#%%
###################
## AIRSPACE
###################

#########################################
## CREATE THE FULLSIZE SEGMENTED STACK ##
#########################################

#Load the binary stack in the original dimensions
binary_zip = zipfile.ZipFile(base_folder_name + sample_name + '/' + binary_filename + '.zip', 'r')
binary_zip.extractall()
binary_raw = binary_zip.open(sample_name + '/' + binary_filename)
binary_stack = io.imread(binary_raw)

io.imshow(binary_stack)

binary_stack = img_as_bool(io.imread(base_folder_name + sample_name + '/' + sample_name + '/' + binary_filename))
#%%
#Check and trim the binary stack if necessary
binary_stack = Trim_Individual_Stack(binary_stack, raw_pred_stack)

# For C_D_5_Strip2_b_
#binary_stack = np.delete(binary_stack, 910, axis=1)
#binary_stack = np.delete(binary_stack, 482, axis=0)

#%%
large_segmented_stack = np.full(shape=binary_stack.shape, fill_value=177, dtype='uint8') # Assign an array filled with the background value 177.
for idx in np.arange(large_segmented_stack.shape[0]):
    temp_veins = img_as_bool(transform.resize(largest_veins[idx],
                                              [binary_stack.shape[1], binary_stack.shape[2]], 
                                              anti_aliasing=False, order=0))    
    temp_epid = transform.resize(unique_epidermis_volumes[idx],
                                              [binary_stack.shape[1], binary_stack.shape[2]], 
                                              anti_aliasing=False, preserve_range=True, order=0) * 30
    leaf_mask = img_as_bool(transform.resize(raw_pred_stack[idx] != bg_value,
                                             [binary_stack.shape[1], binary_stack.shape[2]],
                                             anti_aliasing=False, order=0))
    large_segmented_stack[idx][leaf_mask] = binary_stack[idx][leaf_mask] * 255
    large_segmented_stack[idx][temp_veins] = 147 #vein_value
    large_segmented_stack[idx][temp_epid != 0] = temp_epid[temp_epid != 0]
    
io.imshow(large_segmented_stack[100])
print(np.unique(large_segmented_stack[100]))
io.imsave(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif', large_segmented_stack, imagej=True)

#%%
# Load the large segmeneted stack to re-run the calculations if needed

#large_segmented_stack = io.imread(base_folder_name + sample_name + '/' + sample_name +'SEGMENTED.tif')

#%%
# Redefine the values for the different tissues as used in the segmented image.
# The epidermis will be defined later.
bg_value = 177
spongy_value = 0
palisade_value = 0
if spongy_value == palisade_value:
    mesophyll_value = spongy_value
else:
    mesophyll_value = [spongy_value, palisade_value]
ias_value = 255
vein_value = 147
#%%
epidermis_adaxial_volume = np.sum(large_segmented_stack == abaxial_epidermis_value*30) * vx_volume
epidermis_abaxial_volume = np.sum(large_segmented_stack == adaxial_epidermis_value*30) * vx_volume

epidermis_abaxial_thickness = np.sum(large_segmented_stack == abaxial_epidermis_value*30, axis=1) * px_edge
epidermis_adaxial_thickness = np.sum(large_segmented_stack == adaxial_epidermis_value*60, axis=1) * px_edge

vein_volume = np.sum(large_segmented_stack == 147) * vx_volume
#vein_volume = np.sum(large_segmented_stack == vein_value) * vx_volume


#%%
#Measure the different volumes
leaf_volume = np.sum(large_segmented_stack != bg_value) * vx_volume
mesophyll_volume = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != 30) & (large_segmented_stack != 60)) * vx_volume
cell_volume = np.sum(large_segmented_stack == mesophyll_value) * vx_volume
air_volume = np.sum(large_segmented_stack == ias_value) * vx_volume

print(leaf_volume)
print(cell_volume + air_volume + epidermis_abaxial_volume + epidermis_adaxial_volume + vein_volume)

#%%
#Measure the thickness of the leaf, the epidermis, and the mesophyll
leaf_thickness = np.sum(np.array(large_segmented_stack != bg_value, dtype='bool'), axis=1) * px_edge
mesophyll_thickness = np.sum((large_segmented_stack != bg_value) & (large_segmented_stack != 30) & (large_segmented_stack != 60), axis=1) * px_edge

print(np.median(leaf_thickness),leaf_thickness.mean(),leaf_thickness.std())
print(np.median(mesophyll_thickness),mesophyll_thickness.mean(),mesophyll_thickness.std())

#%%
# Leaf area
leaf_area = large_segmented_stack.shape[0] * large_segmented_stack.shape[2] * (px_edge**2)

#Caluculate Surface Area (adapted from Matt Jenkins' code)
#This gives 1% less surface than from BoneJ's results, but way way faster!!!
ias_vert_faces = marching_cubes_lewiner(large_segmented_stack == ias_value)
ias_SA = mesh_surface_area(ias_vert_faces[0],ias_vert_faces[1])
true_ias_SA = ias_SA * (px_edge**2)
print('IAS surface area: '+str(true_ias_SA)+' µm**2')
print('or '+str(float(true_ias_SA/1000000))+' mm**2')
# end Matt's code

print('Sm: '+str(true_ias_SA/leaf_area))
print('Ames/Vmes: '+str(true_ias_SA/(mesophyll_volume-vein_volume)))

#%%    
# Write the data into a data frame
data_out = {'LeafArea':leaf_area,
            'LeafThickness':leaf_thickness.mean(),
            'LeafThickness_SD':leaf_thickness.std(),
            'MesophyllThickness':mesophyll_thickness.mean(),
            'MesophyllThickness_SD':mesophyll_thickness.std(),
            'ADEpidermisThickness':epidermis_adaxial_thickness.mean(),
            'ADEpidermisThickness_SD':epidermis_adaxial_thickness.std(),
            'ABEpidermisThickness':epidermis_abaxial_thickness.mean(),
            'ABEpidermisThickness_SD':epidermis_abaxial_thickness.std(),
            'LeafVolume':leaf_volume,
            'MesophyllVolume':mesophyll_volume,
            'ADEpidermisVolume':epidermis_adaxial_volume,
            'ABEpidermisVolume':epidermis_abaxial_volume,
            'VeinVolume':vein_volume,
            'CellVolume':cell_volume,
            'IASVolume':air_volume,
            'IASSurfaceArea':true_ias_SA}
results_out = DataFrame(data_out, index={sample_name})
results_out.to_csv(base_folder_name + sample_name + '/' + sample_name + 'RESULTS.txt', sep='\t', encoding='utf-8')
