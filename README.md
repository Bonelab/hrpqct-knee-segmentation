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

This section explains all the steps for generating peri-articular ROIs
for a single longitudinal image series (e.g. the left tibia of one participant).

### 0. Setup

1. Install the environment, as explained above.
2. Clone this repo to your machine (or to ARC).
3. Find the relevant atlases for what you're trying to do. If you want to do the standard peri-articular analysis on knees then you will want the tibia and femur atlases, which should be located on ARC at `/work/boyd_lab/data/Atlases/Knee/periarticular`. If they have moved then email nathan.neeteson@ucalgary.ca or skboyd@ucalgary.ca to ask where to find them. Make sure the atlases each have an image, a mask, and a yaml file.
4. In all further instructions it is implied you have the `blptl` (or whatever you named it) environment activated.

### 1. Data Import

1. Before starting this process, you will need to have generated femur/tibia AIMs from the ISQs using scripts 41 and 42 in `uct_evaluation > Tasks > Evaluation 3D ...`. It would also be best to have already moved your data from the data disk to the projects disk, renamed it, and organized it.
2. Transfer AIMs from the OpenVMS system to your computer (or to ARC) using `ftp` or `sftp`.
3. Convert each AIM to NIfTI format using `python/aim_nifti/convert_aims_to_nifti.py`.

At the end of this step, you should have a directory containing one NIfTI file for each image in the longitudinal series. If you are working in batches you may have one directory for each series, or just one directory with all of the NiFTIs together.

### 2. Baseline ROI Generation

1. Use the `blRegistrationDemons` utility to deformably register the baseline image to the atlas. The corresponding atlas yaml file contains the registration parameters used to generate the atlas and you should use these parameters for the registration as well. Make sure that the atlas is set as the moving image and the baseline image is set as the fixed image, and remember to set the `-mida` flag if the atlas is already downsampled. If you set the `-pmh` optional flag, you will get `png` images saved that show the registration convergence history so you can verify the registration converged properly.
2. Use the `blRegistrationApplyTransform` utility to transform the atlas mask to the image. Make sure that the image is set as the `--fixed-image` optional parameter so that the mask is resampled to the correct physical space and with the correct metadata. Also make sure to set the interpolator as `-int NearestNeighbour` so that the mask labels do not get linearly interpolated.
3. Use the `blVisualizeSegmentation` utility to do a visual check of the atlas-based masks. Registration is finicky, particularly deformable registration. It's possible you may have to go back to step 1 of this section and choose different registration parameters if the registration was botched.

### 3. Follow-up ROI Generation


### 4. Data Export

1. Convert all generated NIfTI masks to AIM files using `python/aim_nifti/convert_mask_to_aim.py`.
2. Transfer AIM masks to the OpenVMS system, using `ftp` or `sftp`, for further processing
