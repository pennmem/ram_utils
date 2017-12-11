import os

from ramutils.tasks import task
from ramutils.reports.generate import ReportGenerator

__all__ = [
    'build_static_report',
]


@task(cache=False)
def build_static_report(session_summaries, math_summaries, delta_hfa_table,
                        classifier_summaries, dest):
    """ Given a set of summary objects, generate a static HTML report """
    generator = ReportGenerator(session_summaries, math_summaries,
                                delta_hfa_table, classifier_summaries,
                                dest=dest)
    report = generator.generate()

    # FIXME: Give this a better name
    final_destination = os.path.join(dest, 'report.html')
    with open(final_destination, 'w') as f:
        f.write(report)

    return True
