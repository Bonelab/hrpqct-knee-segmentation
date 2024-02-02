$! -----------------
$! Get the Arguments
$! -----------------
$!
$   NAME = F$PARSE("''P1'",,,"NAME")
$   DEV      = F$PARSE("''P1'",,,"DEVICE")
$   DIR      = F$PARSE("''P1'",,,"DIRECTORY")
$!
$! -----------------
$!
$ ROI10 := 'DEV''DIR''NAME'_F_ROI10_MASK.AIM
$ ROI11 := 'DEV''DIR''NAME'_F_ROI11_MASK.AIM
$ ROI12 := 'DEV''DIR''NAME'_F_ROI12_MASK.AIM
$ ROI13 := 'DEV''DIR''NAME'_F_ROI13_MASK.AIM
$ ROI14 := 'DEV''DIR''NAME'_F_ROI14_MASK.AIM
$ ROI15 := 'DEV''DIR''NAME'_F_ROI15_MASK.AIM
$ ROI16 := 'DEV''DIR''NAME'_F_ROI16_MASK.AIM
$ ROI17 := 'DEV''DIR''NAME'_F_ROI17_MASK.AIM
$ ROI_F_ALL := 'DEV''DIR''NAME'_F_ROI_ALL_MASK.AIM
$
$ ROI30 := 'DEV''DIR''NAME'_T_ROI30_MASK.AIM
$ ROI31 := 'DEV''DIR''NAME'_T_ROI31_MASK.AIM
$ ROI32 := 'DEV''DIR''NAME'_T_ROI32_MASK.AIM
$ ROI33 := 'DEV''DIR''NAME'_T_ROI33_MASK.AIM
$ ROI34 := 'DEV''DIR''NAME'_T_ROI34_MASK.AIM
$ ROI35 := 'DEV''DIR''NAME'_T_ROI35_MASK.AIM
$ ROI36 := 'DEV''DIR''NAME'_T_ROI36_MASK.AIM
$ ROI37 := 'DEV''DIR''NAME'_T_ROI37_MASK.AIM
$ ROI_T_ALL := 'DEV''DIR''NAME'_T_ROI_ALL_MASK.AIM
$!
$ IPL_BATCH

/db off

/read roi10 "ROI10

/set roi10 10 0

/read roi11 "ROI11

/set roi11 11 0

/read roi12 "ROI12

/set roi12 12 0

/read roi13 "ROI13

/set roi13 13 0

/read roi14 "ROI14

/set roi14 14 0

/read roi15 "ROI15

/set roi15 15 0

/read roi16 "ROI16

/set roi16 16 0

/read roi17 "ROI17

/set roi17 17 0

/add roi10 roi11 roi1011

/add roi12 roi13 roi1213

/add roi14 roi15 roi1415

/add roi16 roi17 roi1617

/add roi1011 roi1213 roi10111213

/add roi1415 roi1617 roi14151617

/add roi10111213 roi14151617 roi_all

/write roi_all "ROI_F_ALL

..
$!
$ IPL_BATCH

/read roi30 "ROI30

/set roi30 30 0

/read roi31 "ROI31

/set roi31 31 0

/read roi32 "ROI32

/set roi32 32 0

/read roi33 "ROI33

/set roi33 33 0

/read roi34 "ROI34

/set roi34 34 0

/read roi35 "ROI35

/set roi35 35 0

/read roi36 "ROI36

/set roi36 36 0

/read roi37 "ROI37

/set roi37 37 0

/add roi30 roi31 roi3031

/add roi32 roi33 roi3233

/add roi34 roi35 roi3435

/add roi36 roi37 roi3637

/add roi3031 roi3233 roi30313233

/add roi3435 roi3637 roi34353637

/add roi30313233 roi34353637 roi_all

/write roi_all "ROI_T_ALL

..
$!
$ EXIT
$!
