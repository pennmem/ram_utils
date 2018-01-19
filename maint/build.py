#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser
import os
import shutil
from subprocess import check_call

parser = ArgumentParser()
parser.add_argument("--no-clean", action="store_true",
                    help="don't remove existing build dir")
parser.add_argument("--python", "-p", nargs="+", default=["2.7", "3.6"],
                    help="python versions to build for (otherwise build all)")


if __name__ == "__main__":
    args = parser.parse_args()

    if not args.no_clean:
        try:
            shutil.rmtree('build')
            os.mkdir('build')
        except OSError:
            pass

    # Extra conda channels to use
    channels = [
        'conda-forge',
        'pennmem',
    ]

    for pyver in args.python:
        build_cmd = [
            "conda", "build",
            "--output-folder=build/",
            "--python", pyver,
        ]

        for chan in channels:
            build_cmd += ['-c', chan]
        build_cmd += ["conda.recipe"]

        print(' '.join(build_cmd))
        check_call(build_cmd)
