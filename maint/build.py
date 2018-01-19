#!/usr/bin/env python

from __future__ import print_function

import os
import shutil
from subprocess import check_call

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

build_cmd = ["conda", "build", "--output-folder=build/"]
for chan in channels:
    build_cmd += ['-c', chan]
build_cmd += ["conda.recipe"]

print(' '.join(build_cmd))
check_call(build_cmd)
