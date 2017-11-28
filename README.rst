ramutils
========

.. image:: https://travis-ci.org/pennmem/ram_utils.svg?branch=master
    :target: https://travis-ci.org/pennmem/ram_utils

.. image:: https://codecov.io/gh/pennmem/ram_utils/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/pennmem/ram_utils

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

Ramulator experiment configuration files for stimulation experiments are
generated via the ``ramulator-conf`` script::

    usage: ramulator-conf [-h] [--root ROOT] [--dest DEST] [--cachedir CACHEDIR]
                      --subject SUBJECT [--force-rerun] --experiment
                      {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}
                      [--vispath VISPATH] [--localization LOCALIZATION]
                      [--montage MONTAGE] --electrode-config-file
                      ELECTRODE_CONFIG_FILE [--anodes ANODES [ANODES ...]]
                      [--cathodes CATHODES [CATHODES ...]]
                      [--min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]]
                      [--max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]]
                      [--target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]]
                      [--pulse-frequencies PULSE_FREQUENCIES [PULSE_FREQUENCIES ...]]

    Generate experiment configs for Ramulator

    optional arguments:
      -h, --help            show this help message and exit
      --root ROOT           path to rhino root
      --dest DEST, -d DEST  directory to write output to
      --cachedir CACHEDIR, -c CACHEDIR
                            absolute path for caching dir
      --subject SUBJECT, -s SUBJECT
                            subject ID
      --force-rerun         force re-running all tasks
      --experiment {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}, -x {AmplitudeDetermination,PS4_FR5,PS4_CatFR5,FR3,CatFR3,PAL3,FR5,CatFR5,PAL5,FR6,CatFR6}
                            experiment
      --vispath VISPATH     path to save task graph visualization to
      --localization LOCALIZATION, -l LOCALIZATION
                            localization number
      --montage MONTAGE, -m MONTAGE
                            montage number
      --electrode-config-file ELECTRODE_CONFIG_FILE, -e ELECTRODE_CONFIG_FILE
                            path to Odin electrode config csv file
      --anodes ANODES [ANODES ...]
                            stim anode labels
      --cathodes CATHODES [CATHODES ...]
                            stim cathode labels
      --min-amplitudes MIN_AMPLITUDES [MIN_AMPLITUDES ...]
                            minimum stim amplitudes
      --max-amplitudes MAX_AMPLITUDES [MAX_AMPLITUDES ...]
                            maximum stim amplitudes
      --target-amplitudes TARGET_AMPLITUDES [TARGET_AMPLITUDES ...], -a TARGET_AMPLITUDES [TARGET_AMPLITUDES ...]
                            target stim amplitudes
      --clear-log           clear the log
