#!/bin/bash
# this script exists because some images had to be recropped and resegmented
./projects/saltacii/shell/segment_image.sh SLTCII0012_FL_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0012_FL_V0S no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0012_FR_V03 yes; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0012_FR_V0S yes; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0013_FR_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0014_FL_V03 yes; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0014_FR_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0015_FL_V02 yes; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0016_FL_V02 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0016_FL_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0016_FR_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0017_FL_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0018_FR_V03 no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0018_FR_V0S no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0023_FR_V0S no; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0055_FR_V02 yes; sleep 0.1
./projects/saltacii/shell/segment_image.sh SLTCII0056_TL_V02 no; sleep 0.1 