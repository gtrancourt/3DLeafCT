# Random Forest and microCT: Leaf traits in 3D

### Random forest segmentation for 3D microCT images
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

Some leaf-traits may be extracted from full stack predictions and/or post-processed full stack predictions. Trait extraction is currently in development stage--see [relevant jupyter notebook](https://github.com/mattjenkins/3DLeafCT/tree/add_changes/ML_microCT/jupyter/LeafTraits.ipynb).


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
#### (oldest)
