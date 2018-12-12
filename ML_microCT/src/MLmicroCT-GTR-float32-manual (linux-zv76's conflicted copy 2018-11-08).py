# Import libraries
import os
import cv2
import numpy as np
import skimage.io as io
from skimage import transform, img_as_int, img_as_ubyte, img_as_float
from skimage.external import tifffile
from skimage.filters import median, sobel, hessian, gabor, gaussian, scharr
from skimage.segmentation import clear_border
from skimage.morphology import cube, ball, disk, remove_small_objects
from skimage.util import invert
import scipy as sp
import scipy.ndimage as spim
import scipy.spatial as sptl
from tabulate import tabulate
import pickle
from PIL import Image
from tqdm import tqdm
from numba import jit
import sklearn as skl
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import misc
from scipy.ndimage.filters import maximum_filter, median_filter, minimum_filter, percentile_filter
from scipy.ndimage.morphology import distance_transform_edt
import vtk
import gc
# Suppress all warnings (not errors) by uncommenting next two lines of code
import warnings
warnings.filterwarnings("ignore")

os.chdir("/home/gtrancourt/Dropbox/_github/3DLeafCT/ML_microCT/src/")
from MLmicroCTfunctions import *


# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152


rescale_factor = 2
Th_grid = 116
Th_phase = 125

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'

folder_name = 'C_D_5_Strip2_b_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'C_D_5_Strip2_b_GRID-8bit-CROPPED.tif'
phase_name = 'C_D_5_Strip2_b_PAGANIN-8bit-CROPPED.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [20,40,60,96,139,176,226,283,321,360,395,416,440,477]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))


gridrec_stack, phaserec_stack, label_stack = Load_images(filepath,grid_name,phase_name,label_name)
#
gridrec_stack, phaserec_stack, label_stack, to_trim = Trimming_Stacks(filepath, gridrec_stack, phaserec_stack, label_stack, rescale_factor, grid_name,phase_name,label_name)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name)

# Generate the local thickness
localthick_stack = localthick_up_save(folder_name, keep_in_memory=True)
localthick_stack = io.imread(folder_name+'local_thick.tif')
#upsample local_thickness images
#localthick_stack = transform.rescale(localthick_stack, rescale_factor)
if rescale_factor > 1:
    localthick_stack = transform.resize(localthick_stack, [localthick_stack.shape[0]*rescale_factor,
                                                           localthick_stack.shape[1]*rescale_factor,
                                                           localthick_stack.shape[2]*rescale_factor])


# Match array dimensions to correct for resolution loss due to downsampling when generating local thickness
gridrec_stack, localthick_stack = match_array_dim(gridrec_stack,localthick_stack)
phaserec_stack, localthick_stack = match_array_dim(phaserec_stack,localthick_stack)
label_stack, localthick_stack = match_array_dim_label(label_stack,localthick_stack)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
#print_feature_layers(rf_transverse,folder_name)

RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[:,:,:],phaserec_stack[:,:,:],localthick_stack[:,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max())RFPredictCTStack_out/255))


#epid_value = 150
#bg_value = 75
#spongy_value = 224
#palisade_value = 224
#ias_value = 0
#vein_value = 92
#
#
#
#step1 = delete_dangling_epidermis(RFPredictCTStack_out,epid_value,bg_value)
#step2 = smooth_epidermis(step1,epid_value,bg_value,spongy_value,palisade_value,ias_value,vein_value)
#processed = final_smooth(step2,vein_value,spongy_value,palisade_value,epid_value,ias_value,bg_value)
#print("Saving post-processed full stack prediction...")
#io.imsave(folder_name+"post_processed_fullstack.tif", img_as_ubyte(processed/255))
#
#

gc.collect()
%reset -f
from MLmicroCTfunctions import *
gc.collect()
# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

rescale_factor = 2
threshold_rescale_factor = 3
Th_grid = 117
Th_phase = 93

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = 'C_D_6_Strip1_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'C_D_6_Strip1_GRID-8bit.tif'
phase_name = 'C_D_6_Strip1_PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [52,111,173,244,332,451,609,728,890,1019,1143,1297,1470,1572,1712,1894]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))

gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor)


# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/GridPhase_invert_ds.tif') == False:
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+'local_thick.tif'):
    localthick_stack = localthick_load_and_resize(folder_name, threshold_rescale_factor)
else:
    localthick_up_save(folder_name, keep_in_memory=False)
    localthick_stack = localthick_load_and_resize(folder_name, threshold_rescale_factor)


# Match array dimensions to correct for resolution loss due to downsampling when generating local thickness
    # GTR: Not needed anymore since the shape has been matched previously
#gridrec_stack, localthick_stack = match_array_dim(gridrec_stack,localthick_stack)
#phaserec_stack, localthick_stack = match_array_dim(phaserec_stack,localthick_stack)
#label_stack, localthick_stack = match_array_dim_label(label_stack,localthick_stack)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
#print_feature_layers(rf_transverse,folder_name)
#save_trainmodel(rf_transverse,FL_train,FL_test,Label_train,Label_test,folder_name)

rf_transverse = load_trainmodel(folder_name)
RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[:,:,:],phaserec_stack[:,:,:],localthick_stack[:,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/255))

gc.collect()











gc.collect()
%reset -f
from MLmicroCTfunctions import *
gc.collect()
# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

rescale_factor = 2
threshold_rescale_factor = 2
Th_grid = 98
Th_phase = 122

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = '_ML_DONE/S_D_2_strip1_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'S_D_2_strip1_GRID-8bit.tif'
phase_name = 'S_D_2_strip1_PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [40, 105, 248, 355, 474, 585, 686, 858, 1062, 1274, 1477, 1632, 1753, 1870]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))

gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor)
#gridrec_stack = Resize_and_save_stack(gridrec_stack, grid_name, rescale_factor, filepath, keep_in_memory=True)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/GridPhase_invert_ds.tif') == False:
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+'local_thick.tif'):
    localthick_stack = io.imread(folder_name+'local_thick.tif')
else:
    localthick_up_save(folder_name, keep_in_memory=False)
    localthick_stack = localthick_load_and_resize(folder_name, threshold_rescale_factor)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
#print_feature_layers(rf_transverse,folder_name)
%reset_selective -f FL_[a-z]
%reset_selective -f Label_[a-z]
gc.collect()
joblib.dump(rf_transverse, folder_name+'RF_model.joblib', compress='zlib')

#rf_transverse = joblib.load(folder_name+'RF_model.joblib')

RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack,phaserec_stack,localthick_stack,"transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/np.max(RFPredictCTStack_out[0])))

tifffile.imsave(folder_name+"test.tif", gridrec_stack, imagej=True)






gc.collect()
%reset -f
from MLmicroCTfunctions import *
gc.collect()
# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

rescale_factor = 2
threshold_rescale_factor = rescale_factor
Th_grid = 105
Th_phase = 95

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = 'S_D_4_strip1_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'S_D_4_Strip1_GRID-8bit.tif'
phase_name = 'S_D_4_Strip1_PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [168, 234, 360, 469, 603, 706, 769, 852, 1029, 1161, 1318, 1467, 1585, 1684, 1843, 1920]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))


gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor)
#gridrec_stack = Resize_and_save_stack(gridrec_stack, grid_name, rescale_factor, filepath, keep_in_memory=True)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/GridPhase_invert_ds.tif') == False:
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+'local_thick.tif'):
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

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
#print_feature_layers(rf_transverse,folder_name)
%reset_selective -f FL_[a-z]
%reset_selective -f Label_[a-z]
gc.collect()
joblib.dump(rf_transverse, folder_name+'RF_model.joblib', compress='zlib')

#rf_transverse = joblib.load(folder_name+'RF_model.joblib')

RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack,phaserec_stack,localthick_stack,"transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))
tifffile.imsave(folder_name+"fullstack_prediction_test.tif", , img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()), imagej=True)






gc.collect()
%reset -f
from MLmicroCTfunctions import *
gc.collect()
# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

rescale_factor = 2
threshold_rescale_factor = 3
Th_grid = 110
Th_phase = 112

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = 'C_I_10_strip2_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'C_I_10_strip2_GRID-8bit.tif'
phase_name = 'C_I_10_strip2_PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [48, 181, 318, 461, 567, 674, 793, 894, 1007, 1181, 1303, 1437, 1559, 1706]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))
gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor)
#gridrec_stack = Resize_and_save_stack(gridrec_stack, grid_name, rescale_factor, filepath, keep_in_memory=True)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile(folder_name+'/GridPhase_invert_ds.tif') == False:
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,threshold_rescale_factor)

# Generate the local thickness
if os.path.isfile(folder_name+'local_thick.tif'):
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

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
#print_feature_layers(rf_transverse,folder_name)
%reset_selective -f FL_[a-z]
%reset_selective -f Label_[a-z]
gc.collect()
joblib.dump(rf_transverse, folder_name+'RF_model.joblib', compress='zlib')

#rf_transverse = joblib.load(folder_name+'RF_model.joblib')

RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack,phaserec_stack,localthick_stack,"transverse")
#RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[0:25,:,:],phaserec_stack[0:25,:,:],localthick_stack[0:25,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/RFPredictCTStack_out.max()))





gc.collect()
%reset -f
from MLmicroCTfunctions import *
gc.collect()
# Define the values of each tissue in the labelled stacks
epid_value = 85
bg_value = 170
spongy_value = 0
palisade_value = 0
ias_value = 255
vein_value = 152

rescale_factor = 2

Th_grid = 90
Th_phase = 129

base_folder_name = '/run/media/gtrancourt/DATADRIVE1/guillaume/_WORK/Vitis/Vitis_greenhouse_shading/microCT/'
folder_name = 'S_I_2_Strip1_/'
if os.path.exists(base_folder_name + folder_name + 'MLresults/') == False:
    os.makedirs(base_folder_name + folder_name + 'MLresults/')

filepath = base_folder_name + folder_name
folder_name = filepath + 'MLresults/'
grid_name = 'S_I_2_Strip1-GRID-8bit.tif'
phase_name = 'S_I_2_Strip1-PAGANIN-8bit.tif'
label_name = 'labelled-stack.tif'

# To take the slices from imageJ notation, put them in python notation (i.e. with 0),
# create a sequence of the same length, shuffle that sequence, and shuffle the slices
# in the same order. This creates a bit of randomness in the training-testing slices.
ImgJ_slices = [62, 130, 279, 415, 534, 705, 892, 980, 1079, 1193, 1307, 1480, 1619, 1862]
labelled_slices = np.array(ImgJ_slices) - 1
labelled_slices_seq = np.arange(labelled_slices.shape[0])
np.random.shuffle(labelled_slices_seq)
labelled_slices = labelled_slices[labelled_slices_seq]
train_slices = np.arange(0,stop=(len(labelled_slices)/2)+1)
test_slices = np.arange((1+len(labelled_slices)/2), stop=len(labelled_slices))

gridrec_stack = Load_Resize_and_Save_Stack(filepath, grid_name, rescale_factor)
phaserec_stack = Load_Resize_and_Save_Stack(filepath, phase_name, rescale_factor)
label_stack = Load_Resize_and_Save_Stack(filepath, label_name, rescale_factor)

# Load the stacks and downsize a copy of the binary to speed up the thickness processing
if os.path.isfile((folder_name+'/GridPhase_invert_ds.tif') == False):
    Threshold_GridPhase_invert_down(gridrec_stack,phaserec_stack,Th_grid,Th_phase,folder_name,rescale_factor)

# Generate the local thickness
if os.path.isfile((folder_name+'local_thick.tif')):
    localthick_stack = io.imread(folder_name+'local_thick.tif')
else:
    localthick_stack = localthick_up_save(folder_name, keep_in_memory=True)

if rescale_factor > 1:
    localthick_stack = transform.resize(localthick_stack, [localthick_stack.shape[0]*rescale_factor,
                                                           localthick_stack.shape[1],
                                                           localthick_stack.shape[2]])

# Match array dimensions to correct for resolution loss due to downsampling when generating local thickness
gridrec_stack, localthick_stack = match_array_dim(gridrec_stack,localthick_stack)
phaserec_stack, localthick_stack = match_array_dim(phaserec_stack,localthick_stack)
label_stack, localthick_stack = match_array_dim_label(label_stack,localthick_stack)

#define image subsets for training and testing
gridphase_train_slices_subset = labelled_slices[train_slices]
gridphase_test_slices_subset = labelled_slices[test_slices]
label_train_slices_subset = labelled_slices_seq[train_slices]
label_test_slices_subset = labelled_slices_seq[test_slices]

displayImages_displayDims(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)

rf_transverse,FL_train,FL_test,Label_train,Label_test = train_model(gridrec_stack,phaserec_stack,label_stack,localthick_stack,gridphase_train_slices_subset,gridphase_test_slices_subset,label_train_slices_subset,label_test_slices_subset)
print('Our Out Of Box prediction of accuracy is: {oob}%'.format(oob=rf_transverse.oob_score_ * 100))
save_trainmodel(rf_transverse,FL_train,FL_test,Label_train,Label_test,folder_name)
#print_feature_layers(rf_transverse,folder_name)

RFPredictCTStack_out = RFPredictCTStack(rf_transverse,gridrec_stack[:,:,:],phaserec_stack[:,:,:],localthick_stack[:,:,:],"transverse")
io.imsave(folder_name+"fullstack_prediction.tif", img_as_ubyte(RFPredictCTStack_out/255))
