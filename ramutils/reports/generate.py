from datetime import datetime
import json
import os.path as osp
import random

from itertools import compress
from jinja2 import Environment, PackageLoader
import numpy as np
from pkg_resources import resource_listdir, resource_string

from ramutils import __version__
from ramutils.reports.summary import FRSessionSummary, MathSummary
from ramutils.events import extract_experiment_from_events, extract_subject
from ramutils.utils import extract_experiment_series


class ReportGenerator(object):
    """Class responsible for generating reports.

    Parameters
    ----------
    session_summaries : List[SessionSummary]
        List of session summaries. The type of report to run is inferred from
        the included summaries.
    math_summaries : List[MathSummary]
        List of math distractor summaries. Can be an empty list for PS stim
        sessions
    sme_table: pd.DataFrame
        Subsequent memory effect table. Can be empty for any stim report
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

    * FR1, catFR1, FR5, catFR5, PS4

    """
    def __init__(self, session_summaries, math_summaries,
                 sme_table, classifier_summaries, dest='.'):
        self.session_summaries = session_summaries
        self.math_summaries = math_summaries
        self.sme_table = sme_table
        self.classifiers = classifier_summaries

        catfr_summary_mask = [summary.experiment == 'catFR1' for summary in
                              self.session_summaries]
        self.catfr_summaries = list(compress(self.session_summaries, catfr_summary_mask))
        self.subject = extract_subject(self.session_summaries[0].events)

        # PS has not math summaries, so only check for non-PS experiments
        if all(['PS' not in exp for exp in self.experiments]):
            if len(session_summaries) != len(math_summaries):
                raise ValueError("Summaries contain different numbers of sessions")

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
        unique_experiments = [np.unique(extract_experiment_from_events(
            summary.events)) for summary in self.session_summaries]
        unique_experiments = np.array(unique_experiments).flatten()
        return unique_experiments

    @property
    def version(self):
        return __version__

    def _make_sme_table(self):
        """ Create data for the SME table for record-only experiments. """
        sme_table = (self.sme_table.sort_values(by='hfa_p_value',
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

    def _make_fr_plot_data(self, joint=False):
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
            'tags': json.dumps([classifier.tag for classifier
                                in self.classifiers])
        }
        if joint:
            plot_data['category'] = json.dumps({
                'irt_between_cat': np.nanmean(np.concatenate(
                    [summary.irt_between_category for summary in
                     self.catfr_summaries])),
                'irt_within_cat': np.nanmean(np.concatenate(
                    [summary.irt_within_category for summary in
                     self.catfr_summaries])),
                'repetition_ratios': self.catfr_summaries[
                    0].repetition_ratios.tolist(),
                'subject_ratio': self.catfr_summaries[0].subject_ratio
            })
        return plot_data

    def generate(self):
        """Central method to generate any report. The report to run is
        determined by the experiments found in :attr:`session_summary`.

        """
        series = extract_experiment_series(self.experiments[0])
        if all(['PS' in exp for exp in self.experiments]) and series == '5':
            return self.generate_ps4_report()

        elif series == '5':
            return self.generate_fr5_report()

        elif all(['FR' in exp for exp in self.experiments]):
            joint = False
            if any(['catFR' in exp for exp in self.experiments]):
                joint = True
            return self.generate_fr_report(joint=joint)

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
            version=self.version,
            subject=self.subject,
            experiment=experiment,
            summaries=self.session_summaries,
            math_summaries=self.math_summaries,
            **kwargs
        )

    def generate_fr_report(self, joint):
        """ Generate an FR1 report

        Returns
        -------
        Rendered FR1 report as a string.

        """
        return self._render(
            'FR1',
            stim=False,
            combined_summary=self._make_combined_summary(),
            classifiers=self.classifiers,
            plot_data=self._make_fr_plot_data(joint=joint),
            sme_table=self._make_sme_table(),
            joint=joint
        )

    def generate_fr5_report(self):
        """ Generate an FR5 report

        Returns
        -------
        Rendered FR5 report as a string.

        """
        return self._render(
            'FR5',
            stim=True,
            combined_summary=self._make_combined_summary(),
            classifiers=self.classifiers,
            stim_params=self.session_summaries[0].stim_parameters,
            recall_tests=self.session_summaries[0].recall_test_results,
            plot_data={
                'roc': json.dumps({
                    'fpr': [classifier.false_positive_rate for classifier in self.classifiers],
                    'tpr': [classifier.true_positive_rate for classifier in self.classifiers],
                }),
                'tercile': json.dumps({
                    'low': [classifier.low_tercile_diff_from_mean for classifier in self.classifiers],
                    'mid': [classifier.mid_tercile_diff_from_mean for classifier in self.classifiers],
                    'high': [classifier.high_tercile_diff_from_mean for classifier in self.classifiers]
                }),
                'tags': json.dumps([classifier.tag for classifier
                                    in self.classifiers]),
                'serialpos': json.dumps({
                    'serialpos': list(range(1, 13)),
                    'overall': {
                        'Overall (non-stim)': self.session_summaries[
                            0].prob_recall_by_serialpos(stim_items_only=False),
                        'Overall (stim)': self.session_summaries[
                            0].prob_recall_by_serialpos(stim_items_only=True)
                    },
                    'first': {
                        'First recall (non-stim)': self.session_summaries[
                            0].prob_first_recall_by_serialpos(stim=False),
                        'First recall (stim)': self.session_summaries[
                            0].prob_first_recall_by_serialpos(stim=True)
                    }
                }),
                'recall_summary': json.dumps({
                    'nonstim': {
                        'listno': self.session_summaries[0].lists(stim=False),
                        'recalled': self.session_summaries[
                            0].recalls_by_list(stim_list_only=False)
                    },
                    'stim': {
                        'listno': self.session_summaries[0].lists(stim=True),
                        'recalled': self.session_summaries[
                            0].recalls_by_list(stim_list_only=True)
                    },
                    'stim_events': {
                        'listno': self.session_summaries[0].lists(),
                        'count': self.session_summaries[0].stim_events_by_list
                    }
                }),
                'stim_probability': json.dumps({
                    'serialpos': list(range(1, 13)),
                    'probability': self.session_summaries[
                        0].prob_stim_by_serialpos
                }),
                'recall_difference': json.dumps({
                    'stim': self.session_summaries[0].delta_recall(),
                    'post_stim': self.session_summaries[0].delta_recall(
                        post_stim_items=True)
                }),
                'classifier_output': json.dumps({
                    'pre_stim': list(self.session_summaries[0].pre_stim_prob_recall),
                    'post_stim': list(self.session_summaries[0].post_stim_prob_recall)
                }),
            }
        )

    def generate_ps4_report(self):
        """ Generate a PS4 report.

        Returns
        -------
        Rendered PS4 report as a string

        """
        location_summary_data = self.session_summaries[0].location_summary
        decision_summary = self.session_summaries[0].decision

        return self._render(
            'PS4',
            stim=True,
            converged=decision_summary['converged'],
            plot_data={
                'ps4': json.dumps(location_summary_data),
            },
            bayesian_optimization_results={
                'best_loc': decision_summary['best_location'],
                'best_ampl': decision_summary['best_amplitude'],
                'p_values': {
                    'between': decision_summary['pval'],
                    'sham': decision_summary['pval_vs_sham']
                },
                'tie': decision_summary['tie'],
                'channels': {
                    channel: {
                        'amplitude': location_summary_data[channel]['best_amplitude'],
                        'delta_classifier': location_summary_data[channel]['best_delta_classifier'],
                        'error': location_summary_data[channel]['sem'],
                        'snr': location_summary_data[channel]['snr']
                    }
                    for channel in list(location_summary_data.keys())
                }
            }
        )
