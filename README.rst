ramutils
========

.. image:: https://travis-ci.org/pennmem/ram_utils.svg?branch=master
    :target: https://travis-ci.org/pennmem/ram_utils

.. image:: https://codecov.io/gh/pennmem/ram_utils/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pennmem/ram_utils

.. image:: https://img.shields.io/badge/docs-here-blue.svg
    :target: https://pennmem.github.io/ram_utils/html/index.html

Bootstrapping a conda environment
---------------------------------

.. code-block:: shell-session

    conda create -y -n environment_name python=3
    source activate environment_name
    conda install -c pennmem ramutils

Usage with the RAM_clinical account
-----------------------------------

For generating Ramulator configuration files, the ``RAM_clinical`` account on
``rhino2`` should be used. This account provides a managed conda environment
and should only be updated by a responsible person from the Systems Development
team.

The general workflow for users is the following:

1. Login to rhino using your normal account
2. Use ``sudo`` to open a shell with the ``RAM_clinical`` user: ``sudo -s -u RAM_clinical``
3. Use ``qlogin`` to open an interactive session on an availabel node: ``qlogin``
4. Activate the ``ramutils`` environment: ``source activate ramutils``
5. Run the provided command line scripts

Ramulator experiment config generation
--------------------------------------

Ramulator experiment configuration files for stimulation experiments are
generated via the ``ramulator-conf`` script::

    usage: ramulator-conf [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                          --subject SUBJECT [--use-cached] --experiment
                          {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}
                          [--vispath VISPATH] [--version]
                          [--localization LOCALIZATION] [--montage MONTAGE]
                          [--electrode-config-file ELECTRODE_CONFIG_FILE]
                          [--anodes ANODES [ANODES ...]]
                          [--cathodes CATHODES [CATHODES ...]]
                          [--min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]]
                          [--max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]]
                          [--target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]]
                          [--no-extended-blanking]
                          [--default-area DEFAULT_AREA | --area-file AREA_FILE]
                          [--clear-log]

    Generate experiment configs for Ramulator

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to (default:
                            scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --use-cached          allow cached results from previous run to be reused
      --experiment {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}, -x {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,FR1,CatFR1,PAL1}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --version             show program's version number and exit
      --localization LOCALIZATION, -l LOCALIZATION
                            localization number (default: 0)
      --montage MONTAGE, -m MONTAGE
                            montage number (default: 0)
      --electrode-config-file ELECTRODE_CONFIG_FILE, -e ELECTRODE_CONFIG_FILE
                            path to existing electrode config CSV file
      --anodes ANODES [ANODES ...], -a ANODES [ANODES ...]
                            stim anode labels
      --cathodes CATHODES [CATHODES ...], -c CATHODES [CATHODES ...]
                            stim cathode labels
      --min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]
                            minimum stim amplitudes
      --max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]
                            maximum stim amplitudes
      --target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...], -t TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]
                            target stim amplitudes
      --no-extended-blanking
                            disable extended blanking
      --default-area DEFAULT_AREA, -A DEFAULT_AREA
                            default surface area to use for all contacts (default:
                            0.001)
      --area-file AREA_FILE
                        path to area.txt file relative to root
  --clear-log           clear the log


Report generation
-----------------

Generating reports on the command line::

    usage: ram-report [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                      --subject SUBJECT [--use-cached] --experiment
                      {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}
                      [--vispath VISPATH] [--version]
                      [--sessions SESSIONS [SESSIONS ...]] [--retrain]
                      [--excluded-contacts EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...]]
                      [--joint-report] [--rerun]
                      [--report_db_location REPORT_DB_LOCATION]
                      [--trigger-electrode TRIGGER_ELECTRODE]

    Generate a report

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to (default:
                            scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --use-cached          allow cached results from previous run to be reused
      --experiment {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}, -x {FR1,CatFR1,PAL1,PS4_FR5,PS4_CatFR5,PS5_FR,PS5_CatFR,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6,AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR6,CatFR6}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --version             show program's version number and exit
      --sessions SESSIONS [SESSIONS ...], -S SESSIONS [SESSIONS ...]
                            sessions to read data from (default: use all)
      --retrain, -R         retrain classifier rather than loading from disk
      --excluded-contacts EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...], -E EXCLUDED_CONTACTS [EXCLUDED_CONTACTS ...]
                            contacts to exclude from classifier
      --joint-report, -j    include CatFR/FR for FR reports (default: off)
      --rerun, -C           do not use previously generated output
      --report_db_location REPORT_DB_LOCATION
                            location of report data database
      --trigger-electrode TRIGGER_ELECTRODE, -t TRIGGER_ELECTRODE
                        Label of the electrode to use for triggering
                        stimulation in PS5




Aggregated Stim Report generation
---------------------------------
Generating an aggregated stim report from the command line::

    usage: ram-aggregated-report [-h] [--root ROOT] [--dest DEST]
                             [--cachedir CACHEDIR] [--use-cached]
                             [--vispath VISPATH] [--version]
                             [--subject SUBJECT [SUBJECT ...]]
                             [--experiment EXPERIMENT [EXPERIMENT ...]]
                             [--sessions SESSIONS [SESSIONS ...]]
                             [--fit-model]
                             [--report_db_location REPORT_DB_LOCATION]

    Generate a report

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root (default: /)
      --dest DEST, -d DEST  directory to write output to (default:
                            scratch/ramutils)
      --cachedir CACHEDIR   absolute path for caching dir
      --use-cached          allow cached results from previous run to be reused
      --vispath VISPATH     path to save task graph visualization to
      --version             show program's version number and exit
      --subject SUBJECT [SUBJECT ...], -s SUBJECT [SUBJECT ...]
                            List of subjects
      --experiment EXPERIMENT [EXPERIMENT ...], -x EXPERIMENT [EXPERIMENT ...]
                            List of experiments
      --sessions SESSIONS [SESSIONS ...], -S SESSIONS [SESSIONS ...]
      --fit-model, -f
      --report_db_location REPORT_DB_LOCATION
                            location of report data database

Info for account maintainers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update an existing conda environment to use the latest release:

``conda update ramutils``

To set up a conda environment for development purposes, execute the following
commands from the top-level of the repository:

.. code-block:: shell-session

    conda create -y -n ramutils_dev python=3
    source activate ramutils_dev
    conda install -c pennmem -c conda-forge --file=requirements.txt

Testing
-------

Automated unit tests that do not require access to RHINO are run with every
push to the remote repository or pull request. Longer running tests requiring
local files can and should be run frequently. Please note that the full test
suite takes appropximately 4 hours to complete. The full test suite should pass
before any new release. Running the full test suite requires two additional
arguments to be passed to pytest::

    --rhino-root: The mount point for RHINO
    --output-dest: Where output from blackbox tests will be saved

You will likely also need to set the environment variable::

    MKL_THREADING_LAYER=GNU

To run the full testing suite from a local copy of the repository, create
a development environment as outlined above, then execute the following from
top level of the repository:

.. code-block:: shell-session

    qlogin
    source activate ramutils_dev
    pytest ramutils/ --rhino-root RHINO_ROOT --output-dest OUTPUT_DEST


The full set of tests are run nightly (5pm EST) from the RAM_maint account on
RHINO by submitting ``ramutils/maint/run_tests.sh`` as a cron job. To update the
timing of the automated tests, switch to the RAM_maint user account on rhino and
edit the crontab file: ``crontab -e``. To update what users receive the
testing report or to modify anything related to the testing itself, update
``ramutils/maint/run_tests.sh``.

