ramutils
========

Installation
------------

Clone the repository::

    $ git clone https://github.com/pennmem/ram_utils.git
    $ cd ram_utils

Create a conda environment with all prerequisites::

    $ conda create -y -n ramutils

Install dependencies::

    $ source activate ramutils
    (ramutils) $ conda install -y -c pennmem --file=requirements.txt

Install ``ramutils``::

    (ramutils) $ python setup.py install


.. note::

    Future releases should be installable via a conda package with dependencies
    specified there.

Contents
--------

.. toctree::
    :maxdepth: 2

    data
    classifier
    events
    pipeline
    cli
    misc
