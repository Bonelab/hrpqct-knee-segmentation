# hrpqct-knee-segmentation
Scripts for training and testing models with the goal of automating knee contouring. 
Depends on `bonelab-pytorch-lightning`

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

| Warning: When setting up an environment, you should install things with `conda` first and then `pip`. <br/>If you flip back and forth you'll end up breaking your environment eventually! |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

## Full Workflow

This section explains all the steps for generating periarticular ROIs
for a single longitudinal image series (e.g. the left tibia of one participant).

It is recommended that you create a main directory to contain the inputs, intermediate outputs, and final outputs for this process. 
In this directory you should have the following sub-directories: `aims`, `niftis`, `atlas_registrations`, `longitudinal_registrations`, `roi_masks`.

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

### 2. Baseline ROI Generation

> **_NOTE:_**  The atlases are created for a RIGHT femur and tibia, respectively. If you want to generate ROIs for a left femur or tibia, there are two extra steps. Before the Demons registration (sub-step 2A below), you need to use `blImageMirror` to mirror your left femur/tibia across axis `0` to create a left-mirrored femur/tibia. Then you can do sub-steps 2A-2C below to get a transformed atlas mask for the left-mirrored femur/tibia. Before moving on to sub-step 2D, you then need to use `blImageMirror` again to recover transformed atlas masks for the original left femur/tibia. Once you do this, you can continue, as normal, with the rest of the steps.

(2A) Use the `blRegistrationDemons` utility to deformably register the baseline image to the atlas. The corresponding atlas yaml file contains the registration parameters used to generate the atlas and you should use these parameters for the registration for consistency. Make sure that the atlas is set as the moving image and the baseline image is set as the fixed image, and remember to set the `-mida` flag if the atlas is already downsampled (it will be, unless you generated your own new atlas). If you set the `-pmh` optional flag, you will get `png` images saved that show the registration convergence history so you can verify the registration converged properly. The registration outputs should go to the `atlas_registrations` subdirectory.

(2B) Use the `blRegistrationApplyTransform` utility to transform the atlas mask to the image. Make sure that the image is set as the `--fixed-image` optional parameter so that the mask is resampled to the correct physical space and with the correct metadata. Also make sure to set the interpolator as `-int NearestNeighbour` so that the mask labels do not get linearly interpolated. The transformed atlas mask should also be output to the `atlas_registrations` subdirectory.

(2C) Use the `blVisualizeSegmentation` utility to visually check the atlas-based masks. Registration is finicky, particularly deformable registration. It's possible you may have to go back to step 1 of this section and choose different registration parameters if the registration was botched.

(2D) .... generate periarticular ROIs using deep learning model(s)

### 3. (Optional) Follow-up ROI Generation

This step only applies if you have a longitudinal (or repeat-measures) dataset. 
If you have a cross-sectional dataset then every image is a "baseline" image and you are done and can skip to the export step.
If you do have a longitudinal (or repeat-measures) dataset, then you need to have completed step 2 on the baseline image before you can process the follow-up images.

> **_NOTE:_**  If you have already longitudinally registered your images then you can skip step 1 below, but instead you will need to use the transformations and common region masks that you already have to take your baseline transformed atlas mask, find its intersection with the common region in the baseline, and then transform that to all follow-ups. This repository does not contain code for doing that as of the writing of these instructions.

(3A) Use the `blRegistrationLongitudinal` utility to rigidly register your follow-up images to the baseline. Send the outputs of the longitudinal registrations to the `longitudinal_registrations` subdirectory. You should pass the transformed atlas baseline mask in and you need to take care to set the various label-related arguments appropriately for how you are naming your images. For example, if I have `sltcii_0002r_t_v0.nii.gz` as my baseline image and `sltcii_0002r_t_v1.nii.gz` as my follow-up image, then I should set `sltcii_0002r_t` as my `output_label`, `v0` as my baseline label, and `v1` as my sole follow-up label. If I then set `atlas_mask` as my mask label, the two masks that get saved will be `sltcii_0002r_t_v0_atlas_mask.nii.gz` and `sltcii_0002r_t_v1_atlas_mask.nii.gz`. Other recommended parameters include: similarity metric `Correlation`, downsampling shrink factor of `4`, downsampling smoothing sigma of `0.5`.

(3B) .... generate periarticular ROIs using deep learning model(s)

### 4. Data Export

(4A) Convert all generated NIfTI ROI masks to AIM files using the `hrpqct-knee-segmentation/python/aim_nifti/convert_mask_to_aim.py` script. Ensure that you set the corresponding original AIM image for each mask so that it gets the correct processing log and metadata and can be read by the VMS system.

(4B) Transfer your periarticular AIM masks to the OpenVMS system, using `ftp` or `sftp`, for further processing.
