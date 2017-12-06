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

        # For debugging/mockups
        self._env.globals['random'] = random

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
        return [np.unique(summary.events.experiment) for summary in self.session_summaries]

    def _make_sme_table(self):
        """Create data for the SME table for record-only experiments.

        FIXME: real data

        """
        return sorted([
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

    def _make_classifier_data(self):
        """Create JSON object for classifier data.

        FIXME: real data

        """
        return {
            'auc': '{:.2f}%'.format(61.35),
            'p_value': '&le; 0.001',
            'output_median': 0.499,
        }

    def generate(self):
        """Central method to generate any report. The report to run is
        determined by the experiments found in :attr:`session_summary`.

        """
        if (np.array(self.experiments) == 'FR1').all():
            return self.generate_fr1_report()
        elif (np.array(self.experiments) == 'FR5').all():
            return self.generate_fr5_report()
        else:
            raise NotImplementedError("Unsupported report type")

    def _render(self, experiment, **kwargs):
        """Convenience method to wrap common keyword arguments passed to the
        template renderer.

        Parameters
        ----------
        experiment : str
        kwargs : dict
            Additional keyword arguments that are passed to the render method.

        """
        template = self._env.get_template(experiment.lower() + '.html')
        return template.render(
            subject=self.subject,
            experiment=experiment,
            summaries=self.session_summaries,
            math_summaries=self.math_summaries,
            classifier=self._make_classifier_data(),
            **kwargs
        )

    def generate_fr1_report(self):
        """Generate an FR1 report.

        Returns
        -------
        Rendered FR1 report as a string.

        """
        return self._render(
            'FR1',
            plot_data={
                'serialpos': json.dumps({
                    'serialpos': list(range(1, 13)),
                    'overall': {
                        'Overall': FRSessionSummary.serialpos_probabilities(self.session_summaries, False),
                    },
                    'first': {
                        'First recall': FRSessionSummary.serialpos_probabilities(self.session_summaries, True),
                    }
                })
            },
            sme_table=self._make_sme_table(),
        )

    def generate_fr5_report(self):
        """Generate an FR5 report.

        Returns
        -------
        Rendered FR5 report as a string.

        """
        return self._render(
            'FR5',
            plot_data={  # FIXME: real data
                'serialpos': json.dumps({
                    'serialpos': list(range(1, 13)),
                    'overall': {
                        'Overall (non-stim)': np.random.random((12,)).tolist(),
                        'Overall (stim)': np.random.random((12,)).tolist()
                    },
                    'first': {
                        'First recall (non-stim)': np.random.random((12,)).tolist(),
                        'First recall (stim)': np.random.random((12,)).tolist()
                    }
                }),
                'recall_summary': json.dumps({
                    'nonstim': {
                        'listno': [4, 8, 11],
                        'recalled': [1, 0, 3]
                    },
                    'stim': {
                        'listno': [5, 6, 7, 9, 10, 12, 13],
                        'recalled': [1, 1, 2, 1, 3, 1, 1]
                    },
                    'stim_events': {
                        'listno': [5, 6, 7, 9, 10, 12, 13],
                        'count': [random.randint(1, 12) for _ in range(7)]
                    }
                })
            }
        )
