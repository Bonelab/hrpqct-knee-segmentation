from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import yaml


def create_parser():
    parser = ArgumentParser(
        description='Copy batch script generation script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "models_file", type=str, metavar="STR",
        help="file path of a csv file containing all of the models to be copied"
    )
    parser.add_argument(
        "directory", type=str, metavar="DIR",
        help="directory to write batch script and yaml file to"
    )
    parser.add_argument(
        "name", type=str, metavar="NAME",
        help="batch script and yaml file name"
    )
    parser.add_argument(
        "--yaml-file", type=str, default=None, metavar="STR",
        help="yaml file to load args from. if given, all other optional args will be ignored."
    )
    parser.add_argument(
        "--preamble", "-p", type=str, metavar="STR",
        default="SUBMIT/QUEUE=SYS$FAST/NOPRINT/NONOTIFY/PRIORITY=30/LOG=SYS$SCRATCH:",
        help="default preamble for batch script"
    )
    parser.add_argument(
        "--script", "-s", type=str, metavar="STR",
        default="DISK4:[BONELAB.PROJECTS.KNEE_AUTOSEG.COM]UTIL_COPY_FILES_PREOA.COM",
        help="default script for batch script"
    )
    parser.add_argument(
        "--log-dir", "-l", type=str, metavar="STR",
        default="DISK4:[BONELAB.PROJECTS.KNEE_AUTOSEG.LOG.PREOA.COPYING]",
        help="default log directory for batch script"
    )
    parser.add_argument(
        "--source-directory", "-s", type=str, metavar="STR",
        default="DISK4:[BONELAB.PROJECTS.PREOA.MODELS]",
        help="directory on the system that files will be copied from"
    )
    parser.add_argument(
        "--target-directory", "-s", type=str, metavar="STR",
        default="DISK4:[BONELAB.PROJECTS.KNEE_AUTOSEG.MODELS.PREOA]",
        help="directory on the system that files will be copied to"
    )
    return parser


def main():
    config = vars(create_parser().parse_args())

    # load yaml file and overwrite the config, if applicable
    if config["yaml_file"] is not None:
        with open(args.yaml_file, 'r') as stream:
            for k, v in yaml.safe_load(stream).items():
                if k != "yaml_file":
                    config[k] = v





if __name__ == "__main__":
    main()