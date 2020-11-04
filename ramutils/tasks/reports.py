from __future__ import unicode_literals

import os
import datetime

from ramutils.tasks import task
from ramutils.reports.generate import ReportGenerator

__all__ = [
    'build_static_report',
]


@task(cache=False)
def build_static_report(subject, experiment, session_summaries, math_summaries,
                        delta_hfa_table, classifier_summaries, dest,
                        hmm_results={}, save=True, aggregated_report=False):
    """ Given a set of summary objects, generate a static HTML report """

    # Subject IDs are at most 8 characters, so this is a quick check to see
    # if multiple subjects are included
    # FIXME: this is amazingly terrible
    if aggregated_report and len(subject) > 8:
        subject = "Multi-subject"


    generator = ReportGenerator(subject, experiment, session_summaries, math_summaries,
                                delta_hfa_table, classifier_summaries,
                                dest=dest, hmm_results=hmm_results)
    report = generator.generate()

    sessions = [str(summary.session_number) for summary in session_summaries]
    sessions_str = "_".join(sessions)

    # For aggregated reports, do not save sessions
    if aggregated_report:
        sessions_str = ""

    today = datetime.datetime.today().strftime('%Y_%m_%d_%H_%M')
    file_name = '_'.join([subject, experiment, sessions_str, today]) + ".html"
    final_destination = os.path.join(dest, file_name)

    if save:
        with open(final_destination, 'w') as f:
            f.write(report)

    return final_destination
