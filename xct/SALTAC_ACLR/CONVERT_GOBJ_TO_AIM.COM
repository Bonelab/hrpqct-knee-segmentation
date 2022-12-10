$ FILENAME = F$PARSE("''P1'",,,"NAME")
$ DEV      = F$PARSE("''P1'",,,"DEVICE")
$ DIR      = F$PARSE("''P1'",,,"DIRECTORY")
$!
$ T_CORT_GOBJ := 'DEV''DIR''FILENAME'_T_CORT.GOBJ
$ T_TRAB_GOBJ := 'DEV''DIR''FILENAME'_T_TRAB.GOBJ
$ F_CORT_GOBJ := 'DEV''DIR''FILENAME'_F_CORT.GOBJ
$ F_TRAB_GOBJ := 'DEV''DIR''FILENAME'_F_TRAB.GOBJ
$ ROI10_GOBJ := 'DEV''DIR''FILENAME'_F_ROI10.GOBJ
$ ROI11_GOBJ := 'DEV''DIR''FILENAME'_F_ROI11.GOBJ
$ ROI12_GOBJ := 'DEV''DIR''FILENAME'_F_ROI12.GOBJ
$ ROI13_GOBJ := 'DEV''DIR''FILENAME'_F_ROI13.GOBJ
$ ROI14_GOBJ := 'DEV''DIR''FILENAME'_F_ROI14.GOBJ
$ ROI15_GOBJ := 'DEV''DIR''FILENAME'_F_ROI15.GOBJ
$ ROI16_GOBJ := 'DEV''DIR''FILENAME'_F_ROI16.GOBJ
$ ROI17_GOBJ := 'DEV''DIR''FILENAME'_F_ROI17.GOBJ
$ ROI30_GOBJ := 'DEV''DIR''FILENAME'_T_ROI30.GOBJ
$ ROI31_GOBJ := 'DEV''DIR''FILENAME'_T_ROI31.GOBJ
$ ROI32_GOBJ := 'DEV''DIR''FILENAME'_T_ROI32.GOBJ
$ ROI33_GOBJ := 'DEV''DIR''FILENAME'_T_ROI33.GOBJ
$ ROI34_GOBJ := 'DEV''DIR''FILENAME'_T_ROI34.GOBJ
$ ROI35_GOBJ := 'DEV''DIR''FILENAME'_T_ROI35.GOBJ
$ ROI36_GOBJ := 'DEV''DIR''FILENAME'_T_ROI36.GOBJ
$ ROI37_GOBJ := 'DEV''DIR''FILENAME'_T_ROI37.GOBJ
$!
$ T_CORT_AIM := 'DEV''DIR''FILENAME'_T_CORT_MASK.AIM
$ T_TRAB_AIM := 'DEV''DIR''FILENAME'_T_TRAB_MASK.AIM
$ T_CORT_AIM := 'DEV''DIR''FILENAME'_F_CORT_MASK.AIM
$ T_TRAB_AIM := 'DEV''DIR''FILENAME'_F_TRAB_MASK.AIM
$ ROI10_AIM := 'DEV''DIR''FILENAME'_F_ROI10.AIM
$ ROI11_AIM := 'DEV''DIR''FILENAME'_F_ROI11.AIM
$ ROI12_AIM := 'DEV''DIR''FILENAME'_F_ROI12.AIM
$ ROI13_AIM := 'DEV''DIR''FILENAME'_F_ROI13.AIM
$ ROI14_AIM := 'DEV''DIR''FILENAME'_F_ROI14.AIM
$ ROI15_AIM := 'DEV''DIR''FILENAME'_F_ROI15.AIM
$ ROI16_AIM := 'DEV''DIR''FILENAME'_F_ROI16.AIM
$ ROI17_AIM := 'DEV''DIR''FILENAME'_F_ROI17.AIM
$ ROI30_AIM := 'DEV''DIR''FILENAME'_T_ROI30.AIM
$ ROI31_AIM := 'DEV''DIR''FILENAME'_T_ROI31.AIM
$ ROI32_AIM := 'DEV''DIR''FILENAME'_T_ROI32.AIM
$ ROI33_AIM := 'DEV''DIR''FILENAME'_T_ROI33.AIM
$ ROI34_AIM := 'DEV''DIR''FILENAME'_T_ROI34.AIM
$ ROI35_AIM := 'DEV''DIR''FILENAME'_T_ROI35.AIM
$ ROI36_AIM := 'DEV''DIR''FILENAME'_T_ROI36.AIM
$ ROI37_AIM := 'DEV''DIR''FILENAME'_T_ROI37.AIM
$!
$ IPL_BATCH

/db_scanco_activate
  -write                     false


/gobj_to_aim
  -gobj_filename             "CORT_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "CORT_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "TRAB_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "TRAB_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI10_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI10_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI11_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI11_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI12_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI12_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI13_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI13_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI14_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI14_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI15_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI15_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI16_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI16_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI17_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI17_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI30_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI30_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI31_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI31_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI32_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI32_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI33_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI33_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI34_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI34_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI35_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI35_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI36_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI36_AIM
  -compress_type             bin
  -version_020               true


/gobj_to_aim
  -gobj_filename             "ROI37_GOBJ
  -output                    in
  -peel_iter                 0

/write_v020
  -name                      in
  -filename                  "ROI37_AIM
  -compress_type             bin
  -version_020               true

..
$!
$ EXIT
$!