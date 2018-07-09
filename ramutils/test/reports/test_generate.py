import pytest
import numpy as np
from pkg_resources import resource_filename
import jinja2
import json


from ramutils.events import extract_biomarker_information
from ramutils.reports.generate import ReportGenerator
from ramutils.reports.summary import (FRSessionSummary, MathSummary,
TICLFRSessionSummary)


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

def test_generate_ticl_report(rhino_root):
    event_file = resource_filename('ramutils.test.test_data',
                                   'ticl_fr_events.npz')
    ticl_events = np.rec.array(np.load(event_file)['events'])
    summary = TICLFRSessionSummary()
    summary.biomarker_events = extract_biomarker_information(ticl_events)
    summary.raw_events = ticl_events[['type', 'stim_list', 'phase']]
    session_summaries = [summary]
    plot_data = json.dumps(
        {'classifier_output': {
        phase: {
            'pre_stim': TICLFRSessionSummary.pre_stim_prob_recall(
                session_summaries, phase),
            'post_stim': TICLFRSessionSummary.all_post_stim_prob_recall(
                session_summaries, phase)}
        for phase in ['ENCODING', 'DISTRACT', 'RETRIEVAL']
        }
    })
    kwargs= dict(experiment='TICL_FR',plot_data=plot_data,summaries=session_summaries)

    env = jinja2.Environment(loader=jinja2.PackageLoader('ramutils.reports',
                                                         'templates')
                             )
    tpl = env.get_template('data_quality.html',)

    _ =tpl.render(**kwargs)

    tpl = env.get_template('ticl_stim_recall_analysis.html')
    tpl.render(**kwargs)




