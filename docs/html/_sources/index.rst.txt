ramutils
========

Installation
------------

Clone the repository::

    $ git clone https://github.com/pennmem/ram_utils.git
    $ cd ram_utils

Create a conda environment with all prerequisites::

    $ conda env create --file test_env.yml -n ramutils

Install in the new environment::

    $ source activate ramutils
    (ramutils) $ python setup.py install

Install remaining requirements::

    (ramutils) $ pip install git+https://github.com/pennmem/classiflib.git
    (ramutils) $ pip install git+https://github.com/pennmem/bptools.git

.. note:: In the future, these packages should be installed automatically with
          conda.

Contents
--------

.. toctree::
    :maxdepth: 2

    data
    classifier
    events
    models
    pipeline
    cli
    misc
