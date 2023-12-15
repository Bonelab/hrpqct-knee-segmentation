$!
$   FILENAME = F$PARSE("''P1'",,,"NAME")
$   DEV      = F$PARSE("''P1'",,,"DEVICE")
$   DIR      = F$PARSE("''P1'",,,"DIRECTORY")
$!
$!
$!
$ ROI_AIM :=        'DEV''DIR''FILENAME'.AIM
$ ROI_MASK :=       'DEV''DIR''FILENAME'_MASK.AIM
$!
$ IPL_BATCH

/db off

/read aim "ROI_AIM

/threshold
  -input                     aim
  -output                    mask
  -lower_in_perm_aut_al      1.000000
  -upper_in_perm_aut_al      100000.000000
  -value_in_range            127
  -unit                      5

/write_v020
  -name                      mask
  -filename                  "ROI_MASK
  -compress_type             bin
  -version_020               true

..
$!
$ EXIT