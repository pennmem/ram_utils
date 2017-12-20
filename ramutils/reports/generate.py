from datetime import datetime
import json
import os.path as osp
import random

from jinja2 import Environment, PackageLoader
import numpy as np
from pkg_resources import resource_listdir, resource_string

from ramutils.reports.summary import FRSessionSummary, MathSummary
from ramutils.events import extract_experiment_from_events


class ReportGenerator(object):
    """Class responsible for generating reports.

    Parameters
    ----------
    session_summaries : List[SessionSummary]
        List of session summaries. The type of report to run is inferred from
        the included summaries.
    math_summaries : List[MathSummary]
        List of math distractor summaries.
    sme_table: pd.DataFrame
        Subsequent memory effect table
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
    def __init__(self, session_summaries, math_summaries,
                 sme_table, classifier_summaries, dest='.'):
        self.session_summaries = session_summaries
        self.math_summaries = math_summaries
        self.sme_table = sme_table
        self.classifiers = classifier_summaries

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

        # Filter to indicate that p-values are small
        self._env.filters['pvalue'] = lambda p: '{:.3f}'.format(p) if p > 0.001 else '&le; 0.001'

        # Give access to some static methods
        self._env.globals['MathSummary'] = MathSummary
        self._env.globals['datetime'] = datetime

        # For debugging/mockups
        self._env.globals['random'] = random

        # Give access to Javascript and CSS sources. When we switch to a
        # non-static reporting format, this will be handled by the web server's
        # static file serving.
        self._env.globals['js'] = {}
        self._env.globals['css'] = {}
        for filename in resource_listdir('ramutils.reports', 'static'):
            if filename.endswith('.js'):
                script = resource_string('ramutils.reports.static', filename).decode()
                self._env.globals['js'][filename.split('.')[0]] = script
            elif filename.endswith('.css'):
                css = resource_string('ramutils.reports.static', filename).decode()
                self._env.globals['css'][filename.split('.')[0]] = css

        self.dest = osp.realpath(osp.expanduser(dest))

    @property
    def experiments(self):
        """Returns a list of experiments found in the session summaries."""
        unique_events = [np.unique(extract_experiment_from_events(
            summary.events)) for summary in self.session_summaries]
        unique_events = np.array(unique_events).flatten()
        return  unique_events

    def _make_sme_table(self):
        """ Create data for the SME table for record-only experiments. """
        sme_table = (self.sme_table.sort_values(by='p_value',
                                                ascending=True)
                         .to_dict(orient='records'))
        return sme_table

    def _make_classifier_data(self):
        """Create JSON object for classifier data """
        return {
            'auc': self.classifier_summary.auc,
            'p_value': self.classifier_summary.pvalue,
            'output_median': self.classifier_summary.median_classifier_output,
        }

    def _make_combined_summary(self):
        """ Aggregate behavioral summary data across given sessions """
        return {
            'n_words': sum([summary.num_words for summary in self.session_summaries]),
            'n_correct': sum([summary.num_correct for summary in self.session_summaries]),
            'n_pli': sum([summary.num_prior_list_intrusions for summary in self.session_summaries]),
            'n_eli': sum([summary.num_extra_list_intrusions for summary in self.session_summaries])
        }

    def generate(self):
        """Central method to generate any report. The report to run is
        determined by the experiments found in :attr:`session_summary`.

        """
        if all(['FR' in exp for exp in self.experiments]):
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
            combined_summary=self._make_combined_summary(),
            classifiers=self.classifiers,
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
                }),
                'roc': json.dumps({
                    'fpr': [classifier.false_positive_rate for classifier in self.classifiers],
                    'tpr': [classifier.true_positive_rate for classifier in self.classifiers],
                }),
                'tercile': json.dumps({
                    'low': [classifier.low_tercile_diff_from_mean for classifier in self.classifiers],
                    'mid': [classifier.mid_tercile_diff_from_mean for classifier in self.classifiers],
                    'high': [classifier.high_tercile_diff_from_mean for classifier in self.classifiers]
                }),
                'tags': json.dumps([classifier.metadata['tag'] for classifier
                                    in self.classifiers])
            },
            sme_table=self._make_sme_table(),
        )

    def generate_fr5_report(self):
        """Generate an FR5 report.

        Returns
        -------
        Rendered FR5 report as a string.

        """
        fake_ps4_data = {
            key: {
                'CH1': np.random.random((100,)).tolist(),
                'CH2': np.random.random((100,)).tolist()
            }
            for key in ['encoding', 'distract', 'retrieval', 'sham', 'post_stim']
        }
        fake_ps4_data['amplitude'] = np.linspace(0, 1, 100).tolist()

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
                }),
                'stim_probability': json.dumps({
                    'serialpos': list(range(1, 13)),
                    'probability': np.random.random((12,)).tolist()
                }),
                'recall_difference': json.dumps({
                    'stim': random.uniform(-60, 60),
                    'post_stim': random.uniform(-60, 60)
                }),
                'classifier_output': json.dumps({
                    'pre_stim': np.random.random((40,)).tolist(),
                    'post_stim': np.random.random((40,)).tolist()
                }),

                # FIXME: only in PS4
                'ps4': json.dumps(fake_ps4_data),
            },

            # FIXME: only in PS4
            bayesian_optimization_results={
                'best_loc': 'LAD1_LAD2',
                'best_ampl': 13.5,
                'p_values': {
                    'between': random.uniform(0.001, 0.05),
                    'sham': random.uniform(0.1, 0.5)
                },
                'tie': random.choice([True, False]),
                'channels': {
                    channel: {
                        'amplitude': random.uniform(0.1, 0.5),
                        'delta_classifier': random.random(),
                        'error': random.uniform(0.001, 1),
                        'snr': random.uniform(0.5, 2.0)
                    }
                    for channel in ['LAD1_LAD2', 'LA11_LA12']
                }
            }
        )
