Changes
=======

Version 2.0.2rc0
----------------

**Preview release**

This version of ramutils enhances Ramulator config generation. Odin ENS
electrode configuration files (both CSV and binary) can now be created by the
pipeline. This eliminates several steps from the workflow for configuring an
experiment. To specify surface areas for contacts, a ``area.txt`` file must
either exist in the same directory as ``jacksheet.txt`` or a path to it can
be specified as a command-line option. See the full documentation for details.

Other changes:

* Record-only experiment configurations (FR1, CatFR1, PAL1) can now be generated
  with the CLI
* Electrode config files can be specified as a command-line option to override
  generating them
* A default value for contact surface areas can be specified in lieu of an area
  file
* The minimum required version of PTSA was bumped up to 1.1.4
* Extended blanking can be toggled with a command-line option
* The script for updating the conda environment was improved


Version 2.0.1
-------------

**2017-12-14**

Version 2.0.1 is a patch to v2.0.0 containing one major bug fix, one minor bug
fix, and other code refactoring that does not alter the behavior of the code.

Summary of changes:

Major Fix: An implicit assumption in the reporting and config generation
pipelines is that the events used to train/evaluate the classifier are in the
same order (sorted by session, list, time) as the rows of the power matrix
(input to the classifier). As part of normalizing the features, the
normalization is done separately for encoding and retrieval events. In v2.0.0,
normalized features were concatenated together without maintaining the original
order. This led to the event order being different from the rows of the power
matrix, resulting in poor classifiers.

Minor Fix: Classifiers trained on encoding and retrieval events should only be
evaluated on out of sample encoding events. In v2.0.0 evaluation was being done
on out of sample encoding and retrieval events.

Reports should continue to be generated with the legacy ramutils code. Other
updates in this patch release include changes that have been made while moving
towards the v2.1 release.


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
