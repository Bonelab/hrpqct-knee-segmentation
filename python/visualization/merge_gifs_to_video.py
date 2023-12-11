from __future__ import annotations

from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from glob import glob
from os import path as ospath
import PIL
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path as plpath

from bonelab.util.echo_arguments import echo_arguments
from bonelab.util.registration_util import check_for_output_overwrite, message_s


def merge_gifs_to_video(args: Namespace) -> None:
    """Merge multiple gifs into a single video."""
    print(echo_arguments("Merge GIFs to Video", vars(args)))
    message_s(f"Looking for GIFs", args.silent)
    gif_list = glob(ospath.join(args.directory, "*.gif"))
    gif_list.sort()
    message_s(f"Found {len(gif_list)} GIFs", args.silent)
    if len(gif_list) == 0:
        raise FileNotFoundError(f"No GIFs found in {args.directory}")
    check_for_output_overwrite(args.output, args.overwrite, args.silent)
    list_of_frames = []
    list_of_fns = []
    for gif_fn in gif_list:
        message_s(f"Reading {gif_fn}", args.silent)
        with PIL.Image.open(gif_fn) as img:
            for i in range(img.n_frames):
                img.seek(i)
                list_of_frames.append(np.array(img))
                list_of_fns.append(plpath(gif_fn).stem)

    fig = plt.figure()
    ax = plt.axes()

    def animate(i: int) -> None:
        ax.clear()
        ax.imshow(list_of_frames[i])
        ax.set_title(list_of_fns[i])
        ax.set_frame_on(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        fig.tight_layout()

    animation_frames = np.arange(len(list_of_frames))
    anim = FuncAnimation(fig, animate, frames=animation_frames, interval=(1/(1000*args.fps)))
    ffmpeg_writer = FFMpegWriter(fps=args.fps)
    anim.save(f"{args.output}", writer=ffmpeg_writer)




def create_parser() -> ArgumentParser:
    """Create the argument parser."""
    parser = ArgumentParser(
        description="Merge multiple gifs into a single video.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "directory",
        help="Input directory with gifs to merge.",
    )
    parser.add_argument(
        "output",
        help="Output video filename.",
    )
    parser.add_argument(
        "-f",
        "--fps",
        help="Frames per second.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "-s",
        "--silent",
        help="Silent mode.",
        action="store_true"
    )
    parser.add_argument(
        "-ow",
        "--overwrite",
        help="Overwrite existing output file.",
        action="store_true"
    )
    return parser


def main() -> None:
    """Run the main function."""
    args = create_parser().parse_args()
    merge_gifs_to_video(args)


if __name__ == "__main__":
    main()