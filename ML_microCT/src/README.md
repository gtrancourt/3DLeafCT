# Random Forest segmentation and plant leaves microCT scans

X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

![alt text][logo]

[logo]: https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/imgs_readme/leaf1.png "translucent epidermis with veins"


This project has been initiated by [Mason Earles](https://github.com/masonearles/3DLeafCT) and took over by [Matt Jenkins](https://github.com/mattjenkins3/3DLeafCT). Since I was actively using it, I tweaked it based on my needs.

[Go here for the version this project is forked from.](https://github.com/mattjenkins3/3DLeafCT)


## Requirements
- __RAM:__ Processing a 5 Gb 8-bit multi-sliced tiff file can peak up to 60 GB of RAM and use up to 30-40 Gb of swap memory (memory written on disk) on Linux (and takes about 3-5 hours to complete). Processing a 250 Mb 8-bit file is of course a lot faster (30-90 minutes) but still takes about 10 Gb of RAM. The programm is memory savvy and this needs to be addressed in future versions in order to tweak the program.
- python 2


## Procedure

### Preparation of leaf microCT images for semi-automatic segmentation
A more detailed exaplanation with images will come.

Briefly, I draw over a few slices, the number of which should be determined for each stack based on quality of the image, venation pattern and quantity, etc. After having created a ROI set for each draw-over slice (i.e. test/training slices), I use a [custom ImageJ macro](https://github.com/gtrancourt/imagej_macros/tree/master/macros). I've created a few over time depending on which tissues I wanted to segment, all named `Batch slice labelled...`. Ask me for which would suit you best and how to edit it. This macro loops over the ROI sets in a folder and creates a labelled stack consisting of the manually segmented tissues painted over the binary image (i.e. the image combining the thresholded gridrec and phase stacks).

### Leaf segmentation: `MLmicroCT-GTR-wrapper.py`
The program is currently setup to run non-interactively from the command line. I chose this in order to run multiple segmentations overnight. Another advantage is that it clears the memory efficiently when the program ends. I do need to give a better name!

Under a Unix or Linux system, the program is called like this:

```
python /path/to/this/repo/3DLeafCT/ML_microCT/src/MLmicroCT-GTR-wrapper.py filename_ PHASE GRID 'list,of,slices,in,imagej,1,2,3,4' rescale_factor threshold_rescale_factor '/path/to/your/image/directory/'
```

Real example:

```
python ~/Dropbox/_github/3DLeafCT/ML_microCT/src/MLmicroCT-GTR-wrapper.py Carundinacea2004_0447_ 82 123 '83,275,321,467,603,692' 1 1 '/run/media/gtrancourt/GTR_Touro/Grasses_uCT/'
```

`python`: This just calls python 2.

`/path/to/this/repo/3DLeafCT/ML_microCT/src/MLmicroCT-GTR-wrapper.py`: This should be the complete path to where the segmentation program is. If you have cloned the repository from github, replace `/path/to/this/repo/` for the path to the repository. This is also the folder in which the functions code is located (`MLmicroCTfunctions.py`) and this file is called by `MLmicroCT-GTR-wrapper.py`.

`filename_`: This the filename and the name of the folder. Right now, it is setup so that the folder and the base file name are exactly the same. By base file name, I mean the first part of your naming convention, like `Carundinacea2004_0447_` which is the name of the folder and also exactly the same as in `Carundinacea2004_0447_GRID-8bit.tif`, the gridrec file name.

`PHASE` and `GRID`: These are the threshold values for the phase contract image (also called paganin reconstruction). Only one value needed.

`'list,of,slices,in,imagej,1,2,3,4'`: This is the list of slices in ImageJ notation, i.e. with 1 being the first element. Needs to be between `''` and separated by commas.

`rescale_factor`: This is a downsizing integer that can be used to resize the stack in order to make the computation faster or to have a file size manageable by the program. It will resize only the _x_ and _y_ axes and so keeps more resolution in the _z_ axis. These files are used during the whole segmentation process. Note that the resulting files will be anisotropic, i.e. one voxel has different dimension in _x_, _y_, and _z_.

`threshold_rescale_factor`: This one resizes _z_, i.e. depth or the slice number, after having resized using the `rescale_factor`. This is used in the computation of the local thickness (computing the largest size within the cells -- used in the random forst segmentation). This is particularly slow process and benefits from a smaller file, and it matters less if there is a loss of resolution in this step. Note that this image is now isotropic, i.e. voxels have same dimensions in all axes.

`'/path/to/your/image/directory/'`: Assuming all your image folder for an experiments are located in the same folder, this is the path to this folder (don't forget the `/` at the end).


### Post-processing and leaf traits analysis
A jupyter notebook rendering of the post-processing and leaf trait analysis code, with some resulting images in the notebook, is available. This code will most probably have to be fine-tuned to each experiment.


## Changes made to the previous version
I've made a lot of small edits here and there that were convenient for my microCT images. I did not change the core of the machine learning function. I did however change how the files were saved, their bit depth, and other features that do affect the memory usage and the speed of computation.

#### Segmentation code
- Split the function and program in two files. `MLmicroCTfunctions.py` is where all the functions are at. This file needs some tidying to remove unnecessary code. `MLmicroCT-GTR-wrapper.py` is the file used to carry out the segmentation. It is used non-interactively, on the command line (see below). I prefered a non-interactive use to simplify my usage of the program and be able to run several segmentations over night. This allows to flush the memory everytime the programm ends, which is essential with the file size used here.
- Runs the segmentation on a downsized stack as my microCT scans were too large to be run in a reasonnable amount of time (i.e. < 6 hours) and with a usable amount of RAM (i.e. not capping 64 Gb of RAM and 64 Gb of swap memory). My scans are 3-5 Gb in size, so the downscale size is ~1 Gb, which works out fine. The resizing factor can be changed to `1` (no resize) by changing the `rescale_factor` value.
- Using now the `resize` instead of `rescale` function in order to resize axes 1 and 2 (x and y in ImageJ). The `rescale` function resized axes 0 and 1 (z and x). It made more sense to me to resize x and y.
- Changed the resize function to use a nearest neighbour interpolation (`order=0` in the function). This prevents the introducion of new values in the predicted stacks, both in the model and final full stack prediction.
- Resized the local thickness file input a second time, this time on axis 0 (z in ImageJ), i.e. the stack used to compute local thickness is now isotropic (all edges of one voxel have the same dimension), which is not the case with the resized images above. This is the binary image generated in function `Threshold_GridPhase_invert_down`. Saves some time in the end, and avoids crashing your computer, even when it has 64 Gb of RAM.
- Added a trimming function so that the original images can be divided by a divider of choice (mainly 2 in my case). This removes one or two pixel-wide rows or columns at the outermost edge (bottom or right on the image). This allows to reuse the same image afterward and to compare the downsized full stack predictions.
- Now automatically saves the random forest model using the `joblib` package. It's fast and a great compression ratio is achieved, so there's no use not to keep it as we can extract information on the model afterwards.
- Saves the local thickness file as 8 bits integer. 256 values are more than enough in my case, but this could easily be switched to 16 bits. Local thickness is only computed as pixel values, so it doesn't matter if there's a lost in resolution.
- Added several steps in the code where, if a file is already present, it will load it instead of generating it. This is useful if the program crashed at one point (i.e. the local thickness file was generated after 3 hours of processing, but something happened to the computer and it crashes in the middle of the model training).
- Randomized the training and testing slices order. I found it preferable to mix up the order. I currently uses half for training, half for testing.
- Uses the ImageJ slice indexing as an input. Avoids having to substract one to all your slice numbers. I generate my labelled stack of training and testing slices using [this ImageJ macro](https://github.com/gtrancourt/imagej_macros/blob/master/macros/Batch%20slice%20labelled%20with%20epidermis%20over%20multiple%20RoiSets.ijm).
- I've removed the computation of the performance metric, beside mentionning the accuracy of the model. Performance metric could be computed later using the saved random forest model (in the `joblib` format).

#### Leaf traits analysis code
This analysis was a work in progress in the previous versions of the code. I have written it from scratch based on my needs. It now computes the following and export the data as a CSV file.

- Thicknesses: Leaf, Epidermis (adaxial and abaxial separately), Mesophyll (everything but the epidermis)
- Volumes: Leaf, Mesophyll, Vein, Cell, Airspace, Epidermis (adaxial and abaxial separately)
- Surface area: Airspace

There is also the possibility to integrate this code with the python version of the procedure used by [Earles et al. 2018](http://www.plantphysiol.org/content/178/1/148). This code can be found in the [following repository](https://github.com/gtrancourt/3DLeafTortuosity). A more final version of this code is needed as it is currently only as a jupyter notebook, and the data isn't exported properly.

## References
__Earles JM, Theroux-Rancourt G, Roddy AB, Gilbert ME, McElrone AJ, Brodersen CR (2018)__ [Beyond Porosity: 3D Leaf Intercellular Airspace Traits That Impact Mesophyll Conductance.](http://www.plantphysiol.org/content/178/1/148) Plant Physiol 178: 148–162.



<!-- ### Random forest segmentation for 3D microCT images
X-ray microcomputed tomography (microCT) is rapidly becoming a popular technique for measuring the 3D geometry of plant organs, such as roots, stems, leaves, flowers, and fruits. Due to the large size of these datasets (> 20 Gb per 3D image), along with the often irregular and complex geometries of many plant organs, image segmentation represents a substantial bottleneck in the scientific pipeline. Here, we are developing a Python module that utilizes machine learning to dramatically improve the efficiency of microCT image segmentation with minimal user input.

## Command Line Execution:

Once installed, ML_microCT can be run from the command line. This version includes both a Manual Mode with user input at multiple points throughout image segmentation process as well as a Read From File Mode, which allow one to execute the entire segmentation process without interruption; even multiple segmentations in desired order.

#### See [ML_microCT_instructions.md] for detailed instructions on running from command line.
[ML_microCT_instructions.md]: https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/ML_microCT/ML_microCT_instructions.md

## Post-processing Beta:
Post-processing of full stack predictions is available in Manual Mode and Read From File Mode. Our pocess removes falsely predicted epidermis, false IAS and mesophyll predictions that fall outside the epidermis, false background predictions that fall inside the epidermis; still relies on hardcoded values for epidermis, background, IAS and palisade/spongy mesophyll--interactives are in the works. Improvements forthcoming.

Once you have a fully post-processed stack, you can generate a smooth 3D mesh of connected 2D shapes, in .stl format. Then, it is possible to smooth surfaces and visualize segmented classes as separate, complementary volumes in 3D space. See image below (captured from .stl viewer in ParaView):

Mesh example:
![alt text][logo]

[logo]: https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/imgs_readme/leaf1.png "translucent epidermis with veins"

Some leaf-traits may be extracted from full stack predictions and/or post-processed full stack predictions. Trait extraction is currently in development stage. See relevant [jupyter notebook](https://github.com/mattjenkins3/3DLeafCT/blob/add_changes/ML_microCT/jupyter/LeafTraits.ipynb).


## Most recent changes:
#### (most recent)
-Removed deprecated files including 'smoot_stl.py' and 'vtk_tif_to_stl.py' in 'ML_microCT/src/' directory

-Forked repository

-Updates to both instructions file and pre-processing file.

-Updates to post-processing algorithm for improved prediction accuracy, overall. Now includes vein corrections.

-Leaf trait measurement jupyter notebook added to 'ML_microCT/jupyter' directory. See here for leaf trait extraction from 3D numpy array data. Traits currently include mesophyll thickness, mesophyll surface area exposed to IAS relative to leaf surface area. Improvements and full integration forthcoming.

-Some leaf trait measurement algorithms, that work on '.stl' meshes, now implemented in 'smoot_stl.py' and 'vtk_tif_to_stl.py' in 'ML_microCT/src/' directory. Improvements and full integration forthcoming.

-Performance metrics can now be calculated in 'manual' mode only. Metrics are available for both unprocessed full stack predictions as well as post-processed full stack predictions. Improvements forthcoming.

-During generation of 2D mesh, you are now prompted with instructions for determining certain pixel values, then you identify these values and export '.stl' files only for desired pixel classes.

-Generation of 2D mesh (.stil files) in manual mode. Currently requires 'smooth.py' and manualy changing lots of hardcoded values.

-Various improvements and bug fixes in both 'manual' and 'file I/O' mode.

-Post-processing is now an optional step in the file I/O method. Post-processing has been updated to include parameters for pixel value of specific classes.

-Now, every time a new scan is segmented all results and data related to this scan will be exported to a new folder with a user-generated name.

-Introduced 'batch run' capability for file I/O method that allows one to segment multiple stacks, using multiple instruction files, without any interruption. Note, batch run will always be slightly risky as the program -may abort at any point, so this is not necessarily a recommended method.
#### (oldest) -->
