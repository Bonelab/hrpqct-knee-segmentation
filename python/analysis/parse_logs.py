from __future__ import annotations

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Any, Hashable, Optional
import numpy as np
import pandas as pd
import os
import torch
import yaml


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description='Log parsing script',
        formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--label-pattern", "-lp", type=str, metavar="LABEL_PATTERN", default="*",
        help="Specify label(s) to parse through as either string or pattern for `glob`."
    )
    parser.add_argument(
        "--output-file", "-o", type=str, metavar="OUTPUT", default="parsed.csv",
        help="Name of csv file to save output to."
    )
    parser.add_argument(
        "--log-dir", "-ld", type=str, metavar="LOG_DIR", default="logs",
        help="Relative path to logs directory you want to parse through."
    )
    parser.add_argument(
        "--only-epoch-metrics", "-em", action="store_true", default=False,
        help="Enable to only gather metrics with `epoch` in the name, helps de-clutter but only works if your task "
             "was logging with `on_epoch=True` "
    )
    parser.add_argument(
        "--cross-validation", "-cv", action="store_true", default=False,
        help="Flag to set if you are parsing logs from cross-validation. If enabled, version sub-directories should "
             "have format: <version>_f<fold>."
    )
    parser.add_argument(
        "--hparams-fn", "-hf", type=str, metavar="STR", default="hparams.yaml",
        help="Name of hparams yaml file in version directories."
    )
    parser.add_argument(
        "--metrics-fn", "-mf", type=str, metavar="STR", default="metrics.csv",
        help="Name of metrics csv file in version directories."
    )
    return parser


def main():
    print("LOG PARSING SCRIPT...")

    # parse args
    args = create_parser().parse_args()

    print('\n'.join(f'{k}: {v}' for k, v in vars(args).items()))

    # glob to find the labels we're going to parse through
    label_dirs = glob(os.path.join(args.log_dir, args.label_pattern))

    print("\nFound the following experiments to parse through:")
    print("\n".join(label_dirs))

    # create list of dataframes to store all data in
    dfs = []

    for label_dir in label_dirs:

        # grab the label string from the path
        label = Path(label_dir).stem

        print(f"\nParsing through: {label_dir}")

        # find all the sub-folders, which are the versions of this experiment with this label
        version_dirs = glob(os.path.join(label_dir, "*"))

        print("\nFound the following versions:")
        print("\n".join(version_dirs))

        print("\nParsing...")

        for version_dir in version_dirs:

            # create a new dataframe for this version
            df = pd.DataFrame()

            # figure out the version and fold identifiers, depends on whether cross-validation was used...
            if args.cross_validation:
                version = Path('_'.join(version_dir.split('_')[:-1])).stem
                fold = version_dir.split('_')[-1][1:]
            else:
                version = Path(version_dir).stem
                fold = None
            print(f"\nVersion: {version}, fold: {fold}")

            # add label, version, and fold to dict
            df["label"] = [label]
            df["version"] = [version]
            df["fold"] = [fold]

            # read the hyperparameters file
            try:
                with open(os.path.join(version_dir, args.hparams_fn), "r") as stream:
                    try:
                        hparams = yaml.load(stream, yaml.CLoader)
                        print('\n'.join(f'{k}: {v}' for k, v in hparams.items()))
                    except yaml.YAMLError as exc:
                        print(exc)
                        print("COULD NOT LOAD HPARAMS, SKIPPING...")
                        continue
            except FileNotFoundError as exc:
                print(exc)
                print("NO HPARAMS FILE, SKIPPING...")
                continue

            # add the hyperparameters to the dict
            for k, v in hparams.items():
                df[f"hparams_{k}"] = [str(v)]

            # read the metrics file
            try:
                metrics = pd.read_csv(os.path.join(version_dir, args.metrics_fn))
            except (FileNotFoundError, pd.errors.EmptyDataError) as e:
                # if the metrics file isn't there, this isn't a version we can analyze
                print("NO METRICS CSV, SKIPPING...")
                continue

            # add the min, max, and final value of the metrics to the dict
            for col in metrics.columns:
                if col not in ["step", "epoch"]:
                    if (not args.only_epoch_metrics) or col.endswith("_epoch"):
                        # terminal output
                        print(f"{col} min: {metrics[col].min()}")
                        print(f"{col} max: {metrics[col].max()}")
                        print(f"{col} final: {metrics[col].dropna(inplace=False).iloc[-1]}")
                        # add to dict
                        df[f"metrics_{col}_min"] = [metrics[col].min()]
                        df[f"metrics_{col}_max"] = [metrics[col].max()]
                        df[f"metrics_{col}_final"] = [metrics[col].dropna(inplace=False).iloc[-1]]

            # add dataframe to list
            dfs.append(df)

    print("\nWriting to file...")

    # finally, concatenate and write the dataframes to a csv
    pd.concat(dfs).to_csv(os.path.join(args.log_dir, args.output_file))

    print("Done.")


if __name__ == "__main__":
    main()
