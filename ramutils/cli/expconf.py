"""Command-line script to generate experimental configuration files for
Ramulator.

"""

import os.path as osp

from ramutils.cli import make_parser, ValidationError, configure_caching
from ramutils.constants import EXPERIMENTS

# Supported experiments
# FIXME: ensure PAL support
experiments = ['AmplitdueDetermination'] + EXPERIMENTS['ps'] + \
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

        min_max_lengths = len(args.anodes) == len(args.min_amplitudes) == len(args.max_amplitudes)
        target_length = len(args.anodes) == len(args.target_amplitudes)
        if not (min_max_lengths or target_length):
            raise ValidationError("Number of stim contacts doesn't match number of amplitude settings")

        if len(args.pulse_frequencies) == 1:
            args.pulse_frequencies = [args.pulse_frequencies[0]] * len(args.anodes)

        if not len(args.pulse_frequencies) == len(args.anodes):
            raise ValidationError("Number of pulse frequencies doesn't match number of stim contacts")


def main():
    from ramutils.parameters import FilePaths, FRParameters
    from ramutils.pipelines.ramulator_config import make_ramulator_config

    args = parser.parse_args()
    validate_stim_settings(args)
    configure_caching(args.dest, args.force_rerun)

    paths = FilePaths(
        root=osp.expanduser(args.root),
        electrode_config_file=osp.expanduser(args.electrode_config_file),
        dest='scratch/ramutils2'  # FIXME: either always use abs paths or always use relpaths
    )

    # FIXME: rhino root?
    paths.pairs = osp.join(paths.root, 'protocols', 'subjects', args.subject,
                           'localizations', args.localization,
                           'montages', args.montage,
                           'neuroradiology', 'current_processed', 'pairs.json')

    # Determine kwargs based on experiment type
    if "FR" in args.experiment:
        params = FRParameters()
    else:
        raise RuntimeError("FIXME: support more than FR")

    # Generate!
    make_ramulator_config(args.subject, args.experiment, paths,
                          args.anodes, args.cathodes, params, args.vispath)


if __name__ == "__main__":
    main()
