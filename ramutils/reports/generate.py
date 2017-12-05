import json
import os.path as osp
import random

from jinja2 import Environment, PackageLoader
import numpy as np
from pkg_resources import resource_listdir, resource_string

from ramutils.reports.summary import FRSessionSummary, MathSummary


class ReportGenerator(object):
    """Class responsible for generating reports.

    Parameters
    ----------
    session_summaries : List[SessionSummary]
        List of session summaries. The type of report to run is inferred from
        the included summaries.
    math_summaries : List[MathSummary]
        List of math distractor summaries.
    dest : str
        Directory to write output to.

    Raises
    ------
    ValueError
        When summaries do not match.

    Notes
    -----
    Session and math summaries are checked for basic consistency:

    * Subject should match
    * Number of summaries should match

    FIXME: check session numbers, experiments match up

    Supported reports:

    * FR1

    """
    def __init__(self, session_summaries, math_summaries, dest='.'):
        self.session_summaries = session_summaries
        self.math_summaries = math_summaries

        if len(session_summaries) != len(math_summaries):
            raise ValueError("Summaries contain different numbers of sessions")

        self.subject = session_summaries[0].events.subject[0]
        for i in range(len(session_summaries)):
            s_subj = session_summaries[i].events.subject == self.subject
            m_subj = math_summaries[i].events.subject == self.subject
            if not (all(s_subj) or all(m_subj)):
                raise ValueError("Subjects should all match")

        self._env = Environment(
            loader=PackageLoader('ramutils.reports', 'templates'),
        )

        # Give access to some static methods
        self._env.globals['MathSummary'] = MathSummary

        # Give access to Javascript sources. When we switch to a non-static
        # reporting format, this will be handled by the web server's static file
        # serving.
        self._env.globals['js'] = {}
        for filename in resource_listdir('ramutils.reports', 'static'):
            script = resource_string('ramutils.reports.static', filename)
            self._env.globals['js'][filename.split('.')[0]] = script

        self.dest = osp.realpath(osp.expanduser(dest))

    @property
    def experiments(self):
        """Returns a list of experiments found in the session summaries."""
        return [np.unique(summary.experiment) for summary in self.session_summaries]

    def generate(self):
        """Central method to generate any report. The report to run is
        determined by the experiments found in :attr:`session_summary`.

        """
        if (np.array(self.experiments) == 'FR1').all():
            return self.generate_fr1_report()
        else:
            raise NotImplementedError("Only FR1 reports are supported so far")

    def generate_fr1_report(self):
        """Generate an FR1 report.

        Returns
        -------
        Rendered FR1 report as a string.

        """
        template = self._env.get_template("fr1.html")

        # FIXME: real values
        sme_table = sorted([
            {
                'type': random.choice(['D', 'G', 'S']),
                'contacts': [random.randint(1, 256) for _ in range(2)],
                'labels': "{}-{}".format("label1", "label2"),
                'atlas_loc': random.choice(['Left', 'Right']) + ' ' + random.choice(['MTL', 'supramarginal', 'fusiform']),
                'p_value': random.uniform(0.0001, 0.1),
                't_stat': random.uniform(-7, 7)
            }
            for _ in range(24)
        ], key=lambda x: x['p_value'])

        return template.render(
            subject=self.subject,
            experiment='FR1',
            summaries=self.session_summaries,
            math_summaries=self.math_summaries,
            plot_data={
                'serialpos': json.dumps({
                    'x': range(1, 13),
                    'y1': FRSessionSummary.serialpos_probabilities(self.session_summaries, False),
                    'y2': FRSessionSummary.serialpos_probabilities(self.session_summaries, True),
                })
            },

            # FIXME: real values
            classifier={
                'auc': '{:.2f}%'.format(61.35),
                'p_value': '&le; 0.001',
                'output_median': 0.499,
            },

            sme_table=sme_table,
        )
