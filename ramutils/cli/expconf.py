"""Command-line script to generate experimental configuration files for
Ramulator.

"""

import os.path as osp

from ramutils.cli import make_parser, ValidationError, configure_caching
from ramutils.constants import EXPERIMENTS

# Supported experiments
# FIXME: ensure PAL support
experiments = ['AmplitudeDetermination'] + EXPERIMENTS['ps'] + \
              [exp for exp in EXPERIMENTS['closed_loop'] if 'PAL' not in exp]

parser = make_parser("Generate experiment configs for Ramulator", experiments)
parser.add_argument('--localization', '-l', default=0, type=int, help='localization number')
parser.add_argument('--montage', '-m', default=0, type=int, help='montage number')
parser.add_argument('--electrode-config-file', '-e', required=True, type=str,
                    help='path to Odin electrode config csv file')
parser.add_argument('--anodes', nargs='+', help='stim anode labels')
parser.add_argument('--cathodes', nargs='+', help='stim cathode labels')
parser.add_argument('--min-amplitudes', nargs='+', help='minimum stim amplitudes')
parser.add_argument('--max-amplitudes', nargs='+', help='maximum stim amplitudes')
parser.add_argument('--target-amplitudes', '-a', nargs='+', help='target stim amplitudes')
parser.add_argument('--pulse-frequencies', '-f', nargs='+', type=float,
                    help='stim pulse frequencies (one to use same value)')


def validate_stim_settings(args):
    """Check stim settings have the right number of arguments."""
    if args.experiment not in ['FR1', 'catFR1']:
        if not len(args.anodes) == len(args.cathodes):
            raise ValidationError("Number of anodes doesn't match number of cathodes")

        if args.experiment != "AmplitudeDetermination":
            if args.target_amplitudes is None:
                raise RuntimeError("--target-amplitudes is required")
            valid = len(args.anodes) == len(args.target_amplitudes)
        else:
            valid = len(args.anodes) == len(args.min_amplitudes) == len(args.max_amplitudes)

        if not valid:
            raise ValidationError("Number of stim contacts doesn't match number of amplitude settings")

        if args.pulse_frequencies is None:
            args.pulse_frequencies = [200] * len(args.anodes)
        elif len(args.pulse_frequencies) == 1:
            args.pulse_frequencies = [args.pulse_frequencies[0]] * len(args.anodes)

        if not len(args.pulse_frequencies) == len(args.anodes):
            raise ValidationError("Number of pulse frequencies doesn't match number of stim contacts")


def main(input_args=None):
    from ramutils.parameters import FilePaths, FRParameters
    from ramutils.pipelines.ramulator_config import make_ramulator_config

    args = parser.parse_args(input_args)
    validate_stim_settings(args)
    configure_caching(osp.join(args.dest, 'cache'), args.force_rerun)

    paths = FilePaths(
        root=osp.expanduser(args.root),
        electrode_config_file=osp.expanduser(args.electrode_config_file),
        dest=args.dest
    )

    paths.pairs = osp.join(paths.root, 'protocols', 'subjects', args.subject,
                           'localizations', str(args.localization),
                           'montages', str(args.montage),
                           'neuroradiology', 'current_processed', 'pairs.json')

    # Determine params based on experiment type
    if args.experiment == 'AmplitudeDetermination':
        params = None
    elif "FR" in args.experiment:
        params = FRParameters()
    else:
        raise RuntimeError("FIXME: support more than FR")

    # Generate!
    make_ramulator_config(args.subject, args.experiment, paths,
                          args.anodes, args.cathodes, params, args.vispath)


if __name__ == "__main__":
    main([
        "-s", "R1364C", "-x", "FR5",
        "-e", "scratch/system3_configs/ODIN_configs/R1364C/R1364C_06NOV2017L0M0STIM.csv",
        "--anodes", "AMY7", "--cathodes", "TOJ8",
        "--target-amplitudes", "0.5",
        "--root", "~/mnt/rhino", "--dest", "scratch/ramutils2/demo", "--force-rerun"
    ])
