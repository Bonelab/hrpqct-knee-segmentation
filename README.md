# hrpqct-knee-segmentation
Scripts for training and testing models with the goal of automating knee contouring. 
Depends on `bonelab-pytorch-lightning`

## Publication

Pre-print: https://doi.org/10.1101/2024.05.20.24307643

This repository contains code for automating peri-articular analysis of bone in knee HR-pQCT images. There are utilities for training, and doing inference with, segmentation models using deep learning (UNet variants specifically), for atlas-based registration, and for ROI generation. This work is not yet peer-reviewed.

## Atlases and trained models

The atlases and trained models are available on zenodo.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11244681.svg)](https://doi.org/10.5281/zenodo.11244681)

## Environment Setup

These instructions are the same as in `bonelab-pytorch-lightning`...

### 1. Set up the recommended `blptl` conda environment:

CPU:
```commandline
conda create -n blptl -c numerics88 -c conda-forge pytorch torchvision pytorch-lightning torchmetrics scikit-learn numpy pandas scipy matplotlib n88tools vtk simpleitk scikit-image
```

GPU:
```commandline
conda create -n blptl -c numerics88 -c conda-forge pytorch-gpu torchvision pytorch-lightning torchmetrics scikit-learn numpy pandas scipy matplotlib n88tools vtk simpleitk scikit-image
```
### 2. Download and install `bonelab`:

```commandline
# from your main projects folder / wherever you keep your git repos...
# ... with SSH authentication
git clone git@github.com:Bonelab/Bonelab.git
# ... or straight HTTPS
git clone https://github.com/Bonelab/Bonelab.git
# go into the repo
cd Bonelab
# install in editable mode
pip install -e .
```

### 3. Download and install `blpytorchlightning`:

```commandline
# from your main projects folder / wherever you keep your git repos...
# ... with SSH authentication
git clone https://github.com/Bonelab/bonelab-pytorch-lightning.git
# ... or straight HTTPS
git clone git@github.com:Bonelab/bonelab-pytorch-lightning.git
# go into the repo
cd bonelab-pytorch-lightning
# install in editable mode
pip install -e .
```

### 4. Install `scikit-image` last
```commandline
pip install scikit-image
```

| Warning: When setting up an environment, you should install things with `conda` first and then `pip`. <br/>If you flip back and forth you'll end up breaking your environment eventually! |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Full Workflow

This section explains all the steps for generating periarticular ROIs
for a single longitudinal image series (e.g. the left tibia of one participant).

It is recommended that you create a main directory to contain the inputs, intermediate outputs, and final outputs for this process. 
In this directory you should have the following sub-directories: `aims`, `niftis`, `atlas_registrations`, `longitudinal_registrations`, `model_masks`, `roi_masks`.

You can organize it differently if you want, but the instructions below assume you have organized it this way.

### 0. Setup

1. Install the environment, as explained above.
2. Clone this repo to your machine (or to ARC).
3. Find the relevant atlases for what you're trying to do. If you want to do the standard periarticular analysis on knees then you will want the tibia and femur atlases with masks for the medial and lateral periarticular contact regions, which should be located on ARC at `/work/boyd_lab/data/Atlases/Knee/periarticular`. If they have moved then email nathan.neeteson@ucalgary.ca or skboyd@ucalgary.ca to ask where to find them. Make sure the atlases each have an image, a mask, and a yaml file.
4. In all further instructions it is assumed that you have the `blptl` (or whatever you named it) environment activated.

### 1. Data Import

(1A) Generate femur/tibia AIMs from the original knee ISQ using scripts 41 and 42 in `uct_evaluation > Tasks > Evaluation 3D ...`. It would also be best to have already moved your data from the data disk to the projects disk, renamed it, and organized it before starting with this procedure.

(1B) Transfer the AIM(s) from the OpenVMS system to the `aims` subdirectory using `ftp` or `sftp`.

(1C) Convert each AIM to NIfTI format using the `hrpqct-knee-segmentation/python/aim_nifti/convert_aims_to_nifti.py` script, and send the output image(s) to the `niftis` subdirectory.

### 2. Segmentation of Subchondral Bone Plate and Trabecular Bone.

(2A) Use `hrpqct-knee-segmentation/python/inference/inference_ensemble.py` to generate an ensembled model predicted segmentation of the subchondral bone plate and trabecular bone from an image nifti. Make sure to set the minimum and maximum density values to the same values as were used to preprocess data when training the models you are using. Outputs should be sent to the `model_masks` subdirectory. You will need some trained knee segmentation models to use this so if you don't know where/how to get some of those, you'll need to ask someone.

(2B) Use `hrpqct-knee-segmentation/python/postprocessing/postprocess_segmentation.py` to post-process the raw model predicted segmentation. The default parameters are recommended, though if you have sub-optimal results with your data you can modify them to see if the output mask improves. Or, you may have a different opinion about the minimum thickness of the subchondral bone plate. Outputs should be sent to the `model_masks` subdirectory.

### 3. Baseline ROI Generation

> **_NOTE:_**  The atlases are created for a RIGHT femur and tibia, respectively. If you want to generate ROIs for a left femur or tibia, there are two extra steps. Before the Demons registration (sub-step 3A below), you need to use `blImageMirror` to mirror your left femur/tibia across axis `0` to create a left-mirrored femur/tibia. Then you can do sub-steps 3A-3C below to get a transformed atlas mask for the left-mirrored femur/tibia. Before moving on to sub-step 3D, you then need to use `blImageMirror` again to recover transformed atlas masks for the original left femur/tibia. Once you do this, you can continue, as normal, with the rest of the steps.

(3A) Use the `blRegistrationDemons` utility to deformably register the baseline image to the atlas. The corresponding atlas yaml file contains the registration parameters used to generate the atlas and you should use these parameters for the registration for consistency. Make sure that the atlas is set as the moving image and the baseline image is set as the fixed image, and remember to set the `-mida` flag if the atlas is already downsampled (it will be, unless you generated your own new atlas). If you set the `-pmh` optional flag, you will get `png` images saved that show the registration convergence history so you can verify the registration converged properly. The registration outputs should go to the `atlas_registrations` subdirectory.

(3B) Use the `blRegistrationApplyTransform` utility to transform the atlas mask to the image. Make sure that the image is set as the `--fixed-image` optional parameter so that the mask is resampled to the correct physical space and with the correct metadata. Also make sure to set the interpolator as `-int NearestNeighbour` so that the mask labels do not get linearly interpolated. The transformed atlas mask should also be output to the `atlas_registrations` subdirectory.

(3C) Use the `blVisualizeSegmentation` utility to visually check the atlas-based masks. Registration is finicky, particularly deformable registration. It's possible you may have to go back to step 1 of this section and choose different registration parameters if the registration was botched.

(3D) Use the `hrpqct-knee-segmentation/python/generate_rois/generate_rois.py` script to generate the periarticular ROIs from the post-processed segmentation mask and atlas mask. You'll need to specify the bone you are processing. The outputs should be sent to the `roi_masks` subdirectory. 

### 4. (Optional) Follow-up ROI Generation

This step only applies if you have a longitudinal (or repeat-measures) dataset. 
If you have a cross-sectional dataset then every image is a "baseline" image and you are done and can skip to the export step.
If you do have a longitudinal (or repeat-measures) dataset, then you need to have completed step 2 on the baseline image before you can process the follow-up images.

> **_NOTE:_**  If you have already longitudinally registered your images then you can skip step 1 below, but instead you will need to use the transformations and common region masks that you already have to take your baseline transformed atlas mask, find its intersection with the common region in the baseline, and then transform that to all follow-ups. This repository does not contain code for doing that as of the writing of these instructions.

(4A) Use the `blRegistrationLongitudinal` utility to rigidly register your follow-up images to the baseline. Send the outputs of the longitudinal registrations to the `longitudinal_registrations` subdirectory. You should pass the transformed atlas baseline mask in and you need to take care to set the various label-related arguments appropriately for how you are naming your images. For example, if I have `sltcii_0002r_t_v0.nii.gz` as my baseline image and `sltcii_0002r_t_v1.nii.gz` as my follow-up image, then I should set `sltcii_0002r_t` as my `output_label`, `v0` as my baseline label, and `v1` as my sole follow-up label. If I then set `atlas_mask` as my mask label, the two masks that get saved will be `sltcii_0002r_t_v0_atlas_mask.nii.gz` and `sltcii_0002r_t_v1_atlas_mask.nii.gz`. Other recommended parameters include: similarity metric `Correlation`, downsampling shrink factor of `4`, downsampling smoothing sigma of `0.5`, shrink factors of `16 8 4 2 1`, smoothing sigmas of `8 4 2 1 0.1`, and a gradient descent learning rate of `0.01`.

(4B) Use `blVisualizeSegmentation` to check the follow-up masks overlaid on the corresponding images. Longitudinal registration can be finicky just like deformable registration, so you may need to iterate on the registration parameters to get good registrations. Make sure you're happy with the transformed masks before moving on to the next sub-step.

(4C) TODO: Create a script for generating follow-up ROIs. The right way to do it is to do the deep learning segmentation on the follow-up image, then use the baseline ROIs and the transformation from follow-up to baseline. Transform the subchondral bone plate segmentation from follow-up to baseline, and use that to modify the shallow ROI and the Sc.BP ROI to properly capture the bone plate and the interface between the bone plate and shallow trab bone. Then transform all of it to the follow-up frame and you have follow-up ROIs that match up to baseline. Probably we want to include Danielle and Matthias' multi-stack registration as an initial step that you should perform with all of your images before doing any analysis in the first place also, to maximize longitudinal registration accuracy.

### 4. Data Export

(4A) Convert all generated NIfTI ROI masks (of the form `*_roi??_mask.nii.gz`) to AIM files using the `hrpqct-knee-segmentation/python/aim_nifti/convert_mask_to_aim.py` script. Ensure that you set the corresponding original AIM image for each mask so that it gets the correct processing log and metadata and can be read by the VMS system.

(4B) Transfer your periarticular AIM masks to the OpenVMS system, using `ftp` or `sftp`, for further processing.
