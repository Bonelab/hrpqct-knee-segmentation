import os
import shutil
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from glob import glob


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Experiment results collation script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        'directory', type=str, metavar='STR',
        help='top-level directory containing logs for all experiments'
    )
    parser.add_argument(
        'output_directory', type=str, metavar='STR',
        help='subdirectory to collate into, within `directory`'
    )
    parser.add_argument(
        '--pattern', '-p', type=str, default='version_*', metavar='STR',
        help='pattern for glob to use to find the experiment version subdirectories'
    )

    return parser


def main():
    args = create_parser().parse_args()

    os.makedirs(
        os.path.join(args.directory, args.output_directory),
        exist_ok=True
    )

    experiment_folders = glob(os.path.join(args.directory, args.pattern))

    for experiment_folder in experiment_folders:
        try:
            experiment_name = os.path.basename(experiment_folder)
            shutil.copyfile(
                os.path.join(experiment_folder, 'metrics.csv'),
                os.path.join(args.directory, args.output_directory, f'{experiment_name}.csv')
            )
            shutil.copyfile(
                os.path.join(experiment_folder, 'hparams.yaml'),
                os.path.join(args.directory, args.output_directory, f'{experiment_name}.yaml')
            )
            print(f'{experiment_name} collated')
        except FileNotFoundError:
            print(f'{experiment_name} couldn\'t be collated')


if __name__ == '__main__':
    main()
