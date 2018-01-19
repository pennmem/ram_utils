#!/usr/bin/env python

from __future__ import print_function

from argparse import ArgumentParser
import glob
import os
import platform
import shlex
import shutil
from subprocess import check_call
import sys

parser = ArgumentParser()
parser.add_argument("--no-clean", action="store_true",
                    help="don't remove existing build dir")
parser.add_argument("--no-build", action="store_true",
                    help="don't build conda packages")
parser.add_argument("--no-convert", action="store_true",
                    help="don't run conda convert")
parser.add_argument("--python", "-p", nargs="+", default=["2.7", "3.6"],
                    help="python versions to build for (otherwise build all)")


def convert():
    """Convert conda packages to other platforms."""
    os_name = {
        'darwin': 'osx',
        'win32': 'win',
        'linux': 'linux'
    }[sys.platform]
    dirname = '{}-{}'.format(os_name, platform.architecture()[0][:2])
    files = glob.glob('build/{}/*.tar.bz2'.format(dirname))

    for filename in files:
        convert_cmd = "conda convert {} -p all -o build/".format(filename)
        print(convert_cmd)
        check_call(shlex.split(convert_cmd))


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

    if not args.no_build:
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

    if not args.no_convert:
        convert()
