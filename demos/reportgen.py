from argparse import ArgumentParser
from functools import partial
import os.path
import webbrowser

import numpy as np
from ptsa.data.readers import BaseEventReader, JsonIndexReader

# from ramutils.constants import EXPERIMENTS
# from ramutils.events import preprocess_events
from ramutils.parameters import FRParameters
from ramutils.reports.generate import ReportGenerator
from ramutils.reports.summary import FRSessionSummary, MathSummary

parser = ArgumentParser(description='generate sample reports')
parser.add_argument('--subject', '-s', required=True, help='subject ID')
parser.add_argument('--experiment', '-x', choices=['FR1', 'FR5'], required=True,
                    help='experiment type', )
parser.add_argument('--root', '-r', default='/', help='data root path')
parser.add_argument('filename', type=str, default=None, help='HTML output filename')


def main():
    args = parser.parse_args()

    jr = JsonIndexReader(os.path.join(args.root, 'protocols', 'r1.json'))
    agg = partial(jr.aggregate_values, subject=args.subject, experiment=args.experiment)

    # task_events = [
    #     BaseEventReader(filename=f).read()
    #     for f in agg('task_events')
    # ]

    # kwargs = FRParameters().to_dict()
    # events = preprocess_events(args.subject, args.experiment,
    #                            kwargs['baseline_removal_start_time'],
    #                            kwargs['retrieval_time'],
    #                            kwargs['empty_epoch_duration'],
    #                            kwargs['pre_event_buf'],
    #                            kwargs['post_event_buf'],
    #                            encoding_only=kwargs['encoding_only'],
    #                            combine_events=kwargs['combine_events'],
    #                            root=args.root)
    # print(events.dtype)

    if args.experiment.startswith('FR'):
        cls = FRSessionSummary
    else:
        raise NotImplementedError("Only FR right now")

    # session_summaries = [cls.create(events[events.session == session])
    #                      for session in np.unique(events.session)][:4]  # cutting off because too many sessions compared to math events for R1111M

    session_summaries = [
        cls.create(BaseEventReader(filename=f).read())
        for i, f in enumerate(agg('task_events'))
        if i < 4
    ]

    math_summaries = [
        MathSummary.create(BaseEventReader(filename=f).read())
        for f in agg('math_events')
    ]

    generator = ReportGenerator(session_summaries, math_summaries)
    report = generator.generate()

    if args.filename is not None:
        with open(args.filename, 'w') as outfile:
            outfile.write(report)
        webbrowser.open('file://' + os.path.abspath(args.filename))
    else:
        print(report)


if __name__ == "__main__":
    main()
