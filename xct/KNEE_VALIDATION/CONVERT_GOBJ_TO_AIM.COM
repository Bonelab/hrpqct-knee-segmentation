$ FILENAME = F$PARSE("''P1'",,,"NAME")
$ DEV      = F$PARSE("''P1'",,,"DEVICE")
$ DIR      = F$PARSE("''P1'",,,"DIRECTORY")
$!
$ CORT_GOBJ := 'DEV''DIR''FILENAME'_CORT.GOBJ
$ TRAB_GOBJ := 'DEV''DIR''FILENAME'_ENDO.GOBJ
$ ROI10_GOBJ := 'DEV''DIR''FILENAME'_ROI10.GOBJ
$ ROI11_GOBJ := 'DEV''DIR''FILENAME'_ROI11.GOBJ
$ ROI12_GOBJ := 'DEV''DIR''FILENAME'_ROI12.GOBJ
$ ROI13_GOBJ := 'DEV''DIR''FILENAME'_ROI13.GOBJ
$ ROI14_GOBJ := 'DEV''DIR''FILENAME'_ROI14.GOBJ
$ ROI15_GOBJ := 'DEV''DIR''FILENAME'_ROI15.GOBJ
$ ROI16_GOBJ := 'DEV''DIR''FILENAME'_ROI16.GOBJ
$ ROI17_GOBJ := 'DEV''DIR''FILENAME'_ROI17.GOBJ
$!
$ CORT_AIM := 'DEV''DIR''FILENAME'_CORT_MASK.AIM
$ TRAB_AIM := 'DEV''DIR''FILENAME'_TRAB_MASK.AIM
$ ROI10_AIM := 'DEV''DIR''FILENAME'_ROI10.AIM
$ ROI11_AIM := 'DEV''DIR''FILENAME'_ROI11.AIM
$ ROI12_AIM := 'DEV''DIR''FILENAME'_ROI12.AIM
$ ROI13_AIM := 'DEV''DIR''FILENAME'_ROI13.AIM
$ ROI14_AIM := 'DEV''DIR''FILENAME'_ROI14.AIM
$ ROI15_AIM := 'DEV''DIR''FILENAME'_ROI15.AIM
$ ROI16_AIM := 'DEV''DIR''FILENAME'_ROI16.AIM
$ ROI17_AIM := 'DEV''DIR''FILENAME'_ROI17.AIM
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

..
$!
$ EXIT
$!