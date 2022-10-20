$! A utility script to *copy* a series of files from one directory to another.
$!
$ IF P2 .EQS. ""
$  THEN
$    WRITE SYS$OUTPUT "================================================================================"
$    WRITE SYS$OUTPUT "A utility script to *copy* a series of files from one directory to another."
$    WRITE SYS$OUTPUT "Can also be used to rename the files at the same time. A typical application"
$    WRITE SYS$OUTPUT "would be to move all the files resulting from a UPAT analysis into a PROJECT"
$    WRITE SYS$OUTPUT "directory."
$    WRITE SYS$OUTPUT ""
$    WRITE SYS$OUTPUT "Requires two parameters: source file, destination file. Adding an extension"
$    WRITE SYS$OUTPUT "is optional because the script will ignore whatever is there and take the root"
$    WRITE SYS$OUTPUT "name and add extensions. The script creates a COM file in the source file root"
$    WRITE SYS$OUTPUT "directory, executes it, and then deletes that file."
$    WRITE SYS$OUTPUT ""
$    WRITE SYS$OUTPUT "Example usage:"
$    WRITE SYS$OUTPUT "@UTILITY_COPY_FILES.COM DISK6:[XTREMECT2.DATA.00000215.00000767]D0000784.AIM -"
$    WRITE SYS$OUTPUT "                DISK4:[BONELAB.PROJECTS.VITD.XT2.MODELS]VITD_0003_RL_M06.AIM"
$    WRITE SYS$OUTPUT ""
$    WRITE SYS$OUTPUT "S Boyd, 2015-05-11"
$    WRITE SYS$OUTPUT "Modified: N Neeteson, 2022-10-18"
$    WRITE SYS$OUTPUT "================================================================================"
$    EXIT
$  ENDIF
$!
$! -----------------------------
$! Get and Define the Arguments
$! -----------------------------
$  FILENAME1 = F$PARSE("''P1'",,,"NAME")
$  DEV1      = F$PARSE("''P1'",,,"DEVICE")
$  DIR1      = F$PARSE("''P1'",,,"DIRECTORY")
$!
$  FILENAME2 = F$PARSE("''P2'",,,"NAME")
$  DEV2      = F$PARSE("''P2'",,,"DEVICE")
$  DIR2      = F$PARSE("''P2'",,,"DIRECTORY")
$!
$!
$	COMMANDFILE := 'DEV1''DIR1'COPY_FILES_'FILENAME1'.COM
$	SHOW SYMBOL COMMANDFILE
$	OPEN/WRITE OUTFILE 'COMMANDFILE'
$	COMMANDFILE = F$SEARCH("''COMMANDFILE'") ! GET VERSION NR
$!
$  IN_FILE          := 'DEV1''DIR1''FILENAME1'
$  OUT_FILE         := 'DEV2''DIR2''FILENAME2'
$!
$!	Write script
$!
$	WRITE OUTFILE "$ ON SEVERE_ERROR THEN CONTINUE"
$	WRITE OUTFILE "$ SET NOON"
$	WRITE OUTFILE "$"
$	WRITE OUTFILE "$ copy "+IN_FILE+".AIM "+OUT_FILE+".AIM"
$	WRITE OUTFILE "$"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_CORT.AIM "+OUT_FILE+"_CORT.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_TRAB.AIM "+OUT_FILE+"_TRAB.AIM"
$	WRITE OUTFILE "$"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI30_MASK.AIM "+OUT_FILE+"_ROI30_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI31_MASK.AIM "+OUT_FILE+"_ROI31_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI32_MASK.AIM "+OUT_FILE+"_ROI32_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI33_MASK.AIM "+OUT_FILE+"_ROI33_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI34_MASK.AIM "+OUT_FILE+"_ROI34_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI35_MASK.AIM "+OUT_FILE+"_ROI35_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI36_MASK.AIM "+OUT_FILE+"_ROI36_MASK.AIM"
$	WRITE OUTFILE "$ copy "+IN_FILE+"_ROI36_MASK.AIM "+OUT_FILE+"_ROI37_MASK.AIM"
$	WRITE OUTFILE "$ EXIT"
$!
$	CLOSE OUTFILE
$!
$! SUBMIT/QUEUE=SYS$FAST/NOPRINT/NONOTIFY/PRIORITY=30/LOG=SYS$SCRATCH: 'COMMANDFILE'
$ @'COMMANDFILE'
$ DELETE 'COMMANDFILE'
$!
$ EXIT