from ramutils.tasks import task

__all__ = [
    'build_static_report',
]


@task()
def build_static_report(session_summaries, math_summaries, classifier_summaries, delta_hfa_table):
    """ Given a set of summary objects, generate a static HTML report """
    return ''
