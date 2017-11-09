#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from ramutils import __version__

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.rst') as history_file:
    history = history_file.read()

# TODO: put package requirements here
requirements = []

# TODO: put setup requirements here
setup_requirements = []

setup(
    name='ramutils',
    version=__version__,
    description="RAM reporting and other utilities",
    long_description=readme + '\n\n' + history,
    author="Penn Computational Memory Lab",
    url='https://github.com/pennmem/ramutils',
    packages=find_packages(include=['ramutils']),
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='ramutils',
    setup_requires=setup_requirements,
)
