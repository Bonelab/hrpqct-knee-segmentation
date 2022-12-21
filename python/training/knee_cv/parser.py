from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace


def create_parser(model_type: str) -> ArgumentParser:
    parser = ArgumentParser(
        description=f'2D, 2.5D, or 3D {model_type} Training Adaptation Version Script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "reference_label", type=str, metavar="REF_LABEL",
        help="label to look in to find the parameters to use. will search in "
             "<log-dir>/<reference_label>/<reference_version>."
    )
    parser.add_argument(
        "reference_version", type=str, metavar="REF_VERSION",
        help="version to look in to find the parameters to use. will search in "
             "<log-dir>/<reference_label>/<reference_version>."
    )
    parser.add_argument(
        "data_dirs", type=str, nargs="+", metavar="DIR",
        help="list of directories to pull data from"
    )
    parser.add_argument(
        "--label", "-l", type=str, default='segan', metavar="STR",
        help="base title to use when saving logs and model checkpoints"
    )
    parser.add_argument(
        "--version", "-v", type=int, default=None, metavar="N",
        help="version number for logging"
    )
    parser.add_argument(
        "--folds", "-f", type=int, default=5, metavar="N",
        help="number of folds to use in cross-validation"
    )
    parser.add_argument(
        "--num-workers", "-w", type=int, default=8, metavar="N",
        help="number of CPU workers to use to load data in parallel"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=1000, metavar="N",
        help="number of epochs to train for"
    )
    parser.add_argument(
        "--batch-size", "-bs", type=int, default=128, metavar="N",
        help='number of samples per minibatch'
    )
    parser.add_argument(
        "--log-dir", "-ld", type=str, default="./logs", metavar="STR",
        help="root directory to store all training logs in"
    )
    parser.add_argument(
        "--log-step-interval", "-lsi", type=int, default=20, metavar="N",
        help="log metrics every N training/validation steps"
    )
    parser.add_argument(
        '--early-stopping-patience', '-esp', type=int, default=40, metavar='N',
        help='number of epochs to train for'
    )
    parser.add_argument(
        "--hours-per-fold", "-hpf", type=int, default=4,
        help="maximum time in hours to spend training the model in each fold"
    )
    parser.add_argument(
        "--cuda", "-c", action="store_true", default=False,
        help="if enabled, check for GPUs and use them"
    )

    return parser