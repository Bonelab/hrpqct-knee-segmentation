$!
$! -----------------
$! Declare segmentation constants
$! -----------------
$! These values are needed for the segmentation portion of the script, they are
$! used with /seg_gauss to threshold and generate the *_SEG.AIM files.
$! Values for these constants were taken from the current default values used
$! in evaluation scripts #29 and #30 in uct_evaluation. It is recommended to
$! not change anything here unless you know what you're doing.
$!
$   SEG_TRAB_SIGMA    :=  0.8
$   SEG_TRAB_SUPPORT  :=  1
$   SEG_TRAB_LOWER    :=  320     ! mg HA/ccm
$   SEG_TRAB_UPPER    :=  3000    ! mg HA/ccm
$   SEG_TRAB_VALUE    :=  127
$   SEG_TRAB_UNIT     :=  2       ! sets units to Density [mg HA/ccm] for lower, upper
$!
$! -----------------
$! Get the Arguments
$! -----------------
$!
$   FILENAME = F$PARSE("''P1'",,,"NAME")
$   DEV      = F$PARSE("''P1'",,,"DEVICE")
$   DIR      = F$PARSE("''P1'",,,"DIRECTORY")
$   ROI_DIR  = "''DIR'" - "]" + ".REF]"
$!
$   ROI_FLAG = P2
$!
$ GREY_AIM :=       'DEV''DIR''FILENAME'.AIM
$ ROI_GOBJ :=       'DEV''ROI_DIR''FILENAME''ROI_FLAG'.GOBJ
$ ROI_AIM :=        'DEV''ROI_DIR''FILENAME''ROI_FLAG'.AIM
$ ROI_MASK :=       'DEV''ROI_DIR''FILENAME''ROI_FLAG'_MASK.AIM
$ ROI_SEG :=        'DEV''ROI_DIR''FILENAME''ROI_FLAG'_SEG.AIM
$ ROI_TH :=         'DEV''ROI_DIR''FILENAME''ROI_FLAG'_TH.AIM
$ ROI_SP :=         'DEV''ROI_DIR''FILENAME''ROI_FLAG'_SP.AIM
$ ROI_NUMBER :=     'DEV''ROI_DIR''FILENAME''ROI_FLAG'_1N.AIM
$!
$!
$!==========================
$! CREATE FILES
$!==========================
$!
$! CREATE ROI_GOBJ from ROI_MASK
$ IPL_BATCH

/read roi_mask "ROI_MASK

/togobj_from_aim roi_mask "ROI_GOBJ 0 0 5

..
$!
$! CREATE ROI_AIM and get scan INFO
$ IPL_BATCH
/read
  -name                      grey
  -filename                  "GREY_AIM

/examine
  -input                     grey
  -item                      log

/gobj_maskaimpeel_ow
  -input_output              grey
  -gobj_filename             "ROI_GOBJ
  -peel_iter                 0

/bounding_box_cut
  -input                     grey
  -output                    bgrey
  -z_only                    false
  -border                    0 0 0

/write_v020
  -name                      bgrey
  -filename                  "ROI_AIM
  -compress_type             bin
  -version_020               true
..
$!
$! CREATE ROI_SEG
$ IPL_BATCH

/read img "GREY_AIM

! add some boundary in x and y (taking dilate1 here) to make sure bone is isolated
! and biggest background piece is connected slicewise AROUND the bone,
/bounding_box_cut
  -input                    img
  -output                   img_box
  -z_only                   false
  -border                   1 1 1

! border in z put into offset then. Will be mirrored later.
/offset_add
  -input                    img_box
  -add_offset               1

/rename
  -old_name                 img_box
  -new_name                 img

! mirror out slices that will be stolen in gauss now:
/fill_offset_mirror
  -input                    img

/sup
  -input                    img
  -supdim_numbers           4 4 1
  -testoff_pixels           1
  -suppos_pixels_local      -1 -1 -1
  -subdim_pixels            -1 -1 -1

/seg_gauss
  -input                    img
  -output                   trabseg
  -sigma                    "SEG_TRAB_SIGMA
  -support                  "SEG_TRAB_SUPPORT
  -low                      "SEG_TRAB_LOWER
  -upp	                    "SEG_TRAB_UPPER
  -value                    "SEG_TRAB_VALUE
  -unit	                    "SEG_TRAB_UNIT

/gobj_maskaimpeel_ow
  -input_output              trabseg
  -gobj_filename             "ROI_GOBJ
  -peel_iter                 0

/bounding_box_cut
  -input                     trabseg
  -output                    trabseg_box
  -z_only                    false
  -border                    0 0 0

/write_v020
  -name                      trabseg_box
  -filename                  "ROI_SEG
  -compress_type             bin
  -version_020               true
..
$ !---------------------------------------------------------
$ ! Tb.Th
$ !---------------------------------------------------------
$ IPL_BATCH

/db_scanco_activate false

/read trab "ROI_SEG

/dt_thickness
  -input                     trab
  -output                    trab_th
  -gobj_filename             "ROI_GOBJ
  -peel_iter                 -1
  -ridge_epsilon             0.900000
  -assign_epsilon            0.500000
  -histofile_or_screen       none
  -suppress_boundary         2
  -version                   3

/write  trab_th    "ROI_TH

..
$ ON ERROR THEN CONTINUE
$ !---------------------------------------------------------
$ ! Tb.Sp
$ !---------------------------------------------------------
$ IPL_BATCH

/db_scanco_activate false

/read trab "ROI_SEG

/dt_spacing
  -input                     trab
  -output                    trab_sp
  -gobj_filename             "ROI_GOBJ
  -peel_iter                 -1
  -ridge_epsilon             0.900000
  -assign_epsilon            0.500000
  -histofile_or_screen       none
  -suppress_boundary         2
  -version                   3

/write  trab_sp    "ROI_SP
..
$ ON ERROR THEN CONTINUE
$ !---------------------------------------------------------
$ ! Tb.N
$ !---------------------------------------------------------
$ IPL_BATCH

/db_scanco_activate false

/read trab "ROI_SEG

/dt_number
  -input                     trab
  -output                    trab_1n
  -gobj_filename             "ROI_GOBJ
  -peel_iter                 -1
  -ridge_epsilon             0.900000
  -assign_epsilon            0.500000
  -histofile_or_screen       none
  -suppress_boundary         2
  -version                   3

/write  trab_1n    "ROI_NUMBER
..
$ ON ERROR THEN CONTINUE
$ !---------------------------------------------------------
$ ! Tb.BMD
$ !---------------------------------------------------------
$ IPL_BATCH

/db_scanco_activate false

/read
  -name                      trab_grey
  -filename                  "ROI_AIM

! trabecular BMD
/voxgobj_scanco_param
  -input                      trab_grey
  -gobj_filename              "ROI_GOBJ
  -peel_iter                  2
  -region                     2
..
$!
$ EXIT
$!
