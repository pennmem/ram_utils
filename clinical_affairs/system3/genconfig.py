# FIXME: generalize this for all experiments (only FR6 now)
# FIXME: handle command-line options and don't prompt for passed options

from __future__ import unicode_literals, print_function

from subprocess import call
from config_prompt import *


def main():
    subject = get_subject()
    experiment = get_experiment(['FR6', 'CatFR6'])
    use_retrieval = get_yes_or_no("Use retrieval data? (y/n) ")
    odin_config_filename = get_path("Odin electrode configuration file: ")

    stim_pairs = []
    for pair in range(2):
        stim_pairs.append(get_stim_pair(experiment, pair))

    cmd = [
        "python",
        "--subject={}".format(subject),
        "--experiment={}".format(experiment),
        "--electrode-config-file={}".format(odin_config_filename),
        "--anodes {}".format(" ".join([pair.anode for pair in stim_pairs])),
        "--cathodes {}".format(" ".join([pair.cathode for pair in stim_pairs])),
        "--pulse-frequency=200",
        "--target-amplitude={}".format(stim_pairs[0].stim_amplitude),  # FIXME for two channels
        "--min-amplitudes {}".format([pair.min_amplitude for pair in stim_pairs]),
        "--max-amplitudes {}".format([pair.max_amplitude for pair in stim_pairs]),
    ]

    if not use_retrieval:
        cmd += ["--encoding-only"]

    print(' '.join(cmd))
    call(cmd)


if __name__ == "__main__":
    main()
