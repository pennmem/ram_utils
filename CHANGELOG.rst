Changes
=======

Version 2.0.0
-------------

**2017-11-30**

Version 2.0 of Ramutils is a major overhaul which restructures the codebase to
improve usability and quality assurance. Common data processing tasks (such as
combining events from different experiments and computing powers) have been
reorganized into reusable and unit-testable functions.

In addition to the restructuring of data processing tasks, this release also
includes a new, unified command line script for generating all stim experiment
configuration files for Ramulator, the RAM System 3 host PC application.

Reports should continue to be generated with the previous version of Ramutils
since the reporting framework has not yet been ported to the restructured
codebase (this is slated for the Ramutils 2.1 release).

Documentation is now available at https://pennmem.github.io/ram_utils/html/index.html.

Summary of changes:

* Restructured for easier mantainability
* Added unit and regression testing
* Added Sphinx documentation
* Unified experiment configuration generation scripts into one entry point
