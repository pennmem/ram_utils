import pytest
import numpy as np
from pkg_resources import resource_filename

from ramutils.reports.generate import ReportGenerator
from ramutils.reports.summary import FRSessionSummary, MathSummary


@pytest.mark.skip(reason='not part of release')
def test_generate_fr1_report():
    """Test that generating the FR1 report runs. It still requires a human to
    check that data is correct.

    """
    events_file = resource_filename(
        'ramutils.test.test_data', 'R1111M_task_events.npz')
    math_events_file = resource_filename(
        'ramutils.test.test_data', 'R1111M_math_events.npz')

    events = np.rec.array(np.load(events_file)['events'])
    math_events = np.rec.array(np.load(math_events_file)['events'])

    session_summaries = []
    math_summaries = []
    for session in np.unique(events.session):
        s_summary = FRSessionSummary()
        s_summary.populate(events[events.session == session])
        m_summary = MathSummary()
        m_summary.populate(math_events[math_events.session == session])

        session_summaries.append(s_summary)
        math_summaries.append(m_summary)

    # import pytest; pytest.set_trace()

    generator = ReportGenerator(session_summaries, math_summaries)
    report = generator.generate_fr1_report()

    with open('out.html', 'w') as rfile:
        rfile.write(report)
