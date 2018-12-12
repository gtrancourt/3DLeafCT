#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 10:07:35 2018

@author: gtrancourt
"""

# Import libraries
import sys
import os
import gc

os.chdir("/home/gtrancourt/Dropbox/_github/3DLeafCT/ML_microCT/src/")
from MLmicroCTfunctions import *

sample_name = sys.argv[1]
Th_phase = int(sys.argv[2])
Th_grid = int(sys.argv[3])
raw_slices = sys.argv[4]
rescale_factor = int(sys.argv[5])
threshold_rescale_factor = int(sys.argv[6])

ImgJ_slices = [int(x) for x in raw_slices.split(',')]

# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = sample_name + '/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name =  sample_name + 'GRID-8bit.tif'
phase_name = sample_name + 'PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))

print('***LOADING IMAGES***')
gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor, labelled_stack=True)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/GridPhase_invert_ds.tif') == False:
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+'local_thick.tif'):
    print('***LOADING LOCAL THICKNESS***')
    localthick_stack = localthick_load_and_resize(folder_name, threshold_rescale_factor)
else:
    localthick_up_save(folder_name, keep_in_memory=False)
    localthick_stack = localthick_load_and_resize(folder_name, threshold_rescale_factor)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)

if os.path.isfile(folder_name+'RF_model.joblib'):
    rf_transverse = joblib.load(folder_name+'RF_model.joblib')
    print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
else:
    #rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
    print('***STARTING MODEL TRAINING***')
    rf_transverse = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
    print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
    gc.collect()
    joblib.dump(rf_transverse, folder_name+'RF_model.joblib', compress='zlib')#print_feature_layers(rf_transverse,folder_name)
    #%reset_selective -f FL_[a-z]
    #%reset_selective -f Label_[a-z]

print('***STARTING FULL STACK PREDICTION***')
RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack,phaserec_stack,localthick_stack,"transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))
print('Done!')
