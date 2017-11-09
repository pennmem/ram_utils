RAM reporting and other utilities
=================================

**Note**: This repo is undergoing a massive cleanup. Things will be
gradually either moved into separate repositories (where common code can
be shared in with other projects) or into a single, top-level
``ramutils`` package.

Installation
------------

Install with ``setup.py``::

    python setup.py install

Ramulator experiment config generation
--------------------------------------

Unified entry point::

    python -m ramutils.cli.expconf
