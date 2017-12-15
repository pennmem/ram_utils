import os
import datetime

from ramutils.tasks import task
from ramutils.reports.generate import ReportGenerator

__all__ = [
    'build_static_report',
]


@task(cache=False)
def build_static_report(subject, experiment, session_summaries, math_summaries,
                        delta_hfa_table, classifier_summaries, dest):
    """ Given a set of summary objects, generate a static HTML report """
    generator = ReportGenerator(session_summaries, math_summaries,
                                delta_hfa_table, classifier_summaries,
                                dest=dest)
    report = generator.generate()

    today = datetime.datetime.today().strftime('%Y_%m_%d')
    file_name = '_'.join([subject, experiment, today]) + ".html"
    final_destination = os.path.join(dest, file_name)

    with open(final_destination, 'w') as f:
        f.write(report)

    return True
