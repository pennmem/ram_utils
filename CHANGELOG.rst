Changes
=======

Upcoming
--------
* Support for building FR2/catFR2, FR3/catFR3, and FR6/catFR6 reports


Version 2.1.4
-------------
**2018-02-14**
* store all underlying data necessary for building reports to facilitate quick reload #118
* do not use cached intermediate results be default (force-rerun)
* clear memory on successful completion of report build
* added montage information (all bipolar pairs, excluded pairs) and normalized powers to serialized data
* added ability to generate a PS5 report


Version 2.1.3
-------------
**2018-01-24**

This is a minor release that includes a few bug fixes discovered when building reports for older subjects as well as a
few minor enhancements. Summary of changes:

* ramutils is now a conda package than can be installed with `conda install -c pennmem ramutils`
* Include the practice list when calculating PLI and ELI.
* Add modal controllability values to target selection table when data is available
* Exclude the first three lists when assessing behavioral response to stim. This was done to bring the current reports
  in line with how these values were reported in the legacy reports
* Automatically detect localization and montage numbers based on the subject, experiment, and session information.
  This still does not handle the case of montage changes from session to session within an experiment
* Allow full test suite to be run from an arbitrary location


Version 2.1.2
-------------

**2018-01-18**

Updated required version of ``bptools`` to 1.3. This fixes a bug that affected
subjects with ECG/EKG channels that appear in the jacksheet prior to the last
set of contacts.


Version 2.1.1
-------------

**2018-01-16**

* Added ``--version`` command-line option (#97)
* Fixed a bug with Ramulator config generation where one step in the pipeline
  was not a task (#100)
* Fixed issue where recarray dtypes were erroneously converted to ``'O'`` type
  meaning they were not portable across Python versions (#101)



Version 2.1.0
-------------

**2018-01-11**

Version 2.1 of ramutils includes report generation using the new dask-based
pipeline framework for the following experiments: FR1, CatFR1, FR5, CatFR5,
PS4_FR5, PS4_CatFR5. Reports are now rendered as static HTML files rather than
PDF and are created using the Jinja templating engine.


Version 2.0.2
-------------

**2017-12-21**

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
* Previews of the new reports (which will officially be rolled out in version
  2.1) are also included


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
