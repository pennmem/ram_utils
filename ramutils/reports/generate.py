from __future__ import unicode_literals

from datetime import datetime
import json
import os.path as osp
import random
import io

from itertools import compress
from jinja2 import Environment, PackageLoader
import numpy as np
from pkg_resources import resource_listdir, resource_string

from ramutils import __version__
from ramutils.reports.summary import FRSessionSummary, MathSummary, FRStimSessionSummary
from ramutils.events import extract_experiment_from_events, extract_subject
from ramutils.utils import extract_experiment_series
from ramutils.tasks.misc import encode_file

class ReportGenerator(object):
    """Class responsible for generating reports.

    Parameters
    ----------
    subject: str
    experiment: str
    session_summaries : List[SessionSummary]
        List of session summaries. The type of report to run is inferred from
        the included summaries.
    math_summaries : List[MathSummary]
        List of math distractor summaries. Can be an empty list for PS stim
        sessions
    sme_table: pd.DataFrame
        Subsequent memory effect table. Can be empty for any stim report
    classifier_summaries: List[ClassifierSummary]
        List of classifier performance summaries.
    hmm_results: Dict
        Dictionary describing the HMM-estimated effects of stimulation (optional)
    dest : str
        Directory to write output to (optional; defaults to the current directory).

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

    * FR1, catFR1, FR2, catFR2, FR3, catFR3, FR5, catFR5, PS4

    """
    def __init__(self, subject, experiment, session_summaries, math_summaries,
                 sme_table, classifier_summaries, hmm_results=None, dest='.'):
        self.subject = subject
        self.experiment = experiment
        self.session_summaries = session_summaries
        self.math_summaries = math_summaries
        self.sme_table = sme_table
        self.classifier_summaries = classifier_summaries

        catfr_summary_mask = [summary.experiment == 'catFR1' for summary in
                              self.session_summaries]
        self.catfr_summaries = list(compress(self.session_summaries, catfr_summary_mask))
        self.hmm_results = hmm_results

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

    def _make_feature_plots(self):
        def get_feature_plot(s):
            fd = io.BytesIO()
            s.plot_normalized_powers(fd)
            return encode_file(fd)

        feature_data = {
            'feature_data': [summary.normalized_powers.tolist() for summary in self.session_summaries],
            'feature_plots': [get_feature_plot(summary) for summary in self.session_summaries]
        }
        return feature_data

    def _make_combined_summary(self):
        """ Aggregate behavioral summary data across given sessions """
        return {
            'n_words': sum([summary.num_words for summary in self.session_summaries]),
            'n_correct': sum([summary.num_correct for summary in self.session_summaries]),
            'n_pli': sum([summary.num_prior_list_intrusions for summary in self.session_summaries]),
            'n_eli': sum([summary.num_extra_list_intrusions for summary in self.session_summaries])
        }

    def _make_plot_data(self, stim=False, classifier=False, joint=False, biomarker_delta=False):
        """ Build up a large dictionary of data for various plots from plot-specific components """

        plot_data = {}
        if not stim:
            plot_data['serialpos'] = {
                'serialpos': list(range(1, 13)),
                'overall': {
                    'Overall': FRSessionSummary.serialpos_probabilities(self.session_summaries, first=False),
                },
                'first': {
                    'First recall': FRSessionSummary.serialpos_probabilities(self.session_summaries, first=True),
                }
            }
            # Only non-stim reports have the option of this IRT plot
            if joint:
                plot_data['category'] = {
                    'irt_between_cat': np.nanmean(np.concatenate(
                        [summary.irt_between_category for summary in
                         self.catfr_summaries])),
                    'irt_within_cat': np.nanmean(np.concatenate(
                        [summary.irt_within_category for summary in
                         self.catfr_summaries])),
                    'repetition_ratios': self.catfr_summaries[
                        0].repetition_ratios.tolist(),
                    'subject_ratio': self.catfr_summaries[0].subject_ratio
                }
        else:
            plot_data['serialpos'] = {
                    'serialpos': list(range(1, 13)),
                    'overall': {
                        'Overall (non-stim)': FRStimSessionSummary.prob_recall_by_serialpos(self.session_summaries,
                                                                                            stim_items_only=False),
                        'Overall (stim)': FRStimSessionSummary.prob_recall_by_serialpos(self.session_summaries,
                                                                                        stim_items_only=True)
                    },
                    'first': {
                        'First recall (non-stim)': FRStimSessionSummary.prob_first_recall_by_serialpos(self.session_summaries,
                                                                                                       stim=False),
                        'First recall (stim)': FRStimSessionSummary.prob_first_recall_by_serialpos(self.session_summaries,
                                                                                                   stim=True)
                    }
                }
            plot_data['recall_summary'] = {
                    'nonstim': {
                        'listno': FRStimSessionSummary.lists(self.session_summaries, stim=False),
                        'recalled': FRStimSessionSummary.recalls_by_list(self.session_summaries, stim_list_only=False)
                    },
                    'stim': {
                        'listno': FRStimSessionSummary.lists(self.session_summaries, stim=True),
                        'recalled': FRStimSessionSummary.recalls_by_list(self.session_summaries, stim_list_only=True)
                    },
                    'stim_events': {
                        'listno': FRStimSessionSummary.lists(self.session_summaries),
                        'count': FRStimSessionSummary.stim_events_by_list(self.session_summaries)
                    }
                }
            plot_data['stim_probability'] = {
                    'serialpos': list(range(1, 13)),
                    'probability': FRStimSessionSummary.prob_stim_by_serialpos(self.session_summaries)
                }
            plot_data['recall_difference'] = {
                    'stim': FRStimSessionSummary.delta_recall(self.session_summaries),
                    'post_stim': FRStimSessionSummary.delta_recall(self.session_summaries, post_stim_items=True)
                }

        if biomarker_delta:
            plot_data['classifier_output'] = {
                'pre_stim': FRStimSessionSummary.pre_stim_prob_recall(self.session_summaries),
                'post_stim': FRStimSessionSummary.all_post_stim_prob_recall(self.session_summaries)
            }

        if classifier:
            plot_data['roc'] = {
                'fpr': [classifier.false_positive_rate for classifier in self.classifier_summaries],
                'tpr': [classifier.true_positive_rate for classifier in self.classifier_summaries],
            }
            plot_data['tercile'] = {
                'low': [classifier.low_tercile_diff_from_mean for classifier in self.classifier_summaries],
                'mid': [classifier.mid_tercile_diff_from_mean for classifier in self.classifier_summaries],
                'high': [classifier.high_tercile_diff_from_mean for classifier in self.classifier_summaries]
            }
            plot_data['tags'] = [classifier.id for classifier in self.classifier_summaries]
            plot_data['weights'] = [encode_file(classifier.plot_classifier_weights())
                                    for classifier in self.classifier_summaries]

        return json.dumps(plot_data)

    def generate(self):
        """Central method to generate any report. The report to run is
        determined by the experiments found in :attr:`session_summary`.

        """
        series = extract_experiment_series(self.experiment)
        if all(['PS4' in exp for exp in self.experiments]) and series == '5':
            return self.generate_ps4_report()

        elif all(['PS5' in exp for exp in self.experiments]):
            return self.generate_ps5_report()

        elif series == '1':
            joint = False
            if any(['catFR' in exp for exp in self.experiments]):
                joint = True
            return self.generate_record_only_report(joint=joint)

        elif series == '2':
            return self.generate_open_loop_fr_report()

        elif series == '3':
            return self.generate_closed_loop_fr_report('FR3')

        elif series == '5':
            return self.generate_closed_loop_fr_report('FR5')

        elif series == '6':
            return self.generate_closed_loop_fr_report('FR6')

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

    def generate_record_only_report(self, joint):
        """ Generate report for record only experiments

        Returns
        -------
        Rendered FR1 report as a string.

        """
        return self._render(
            'FR1',
            stim=False,
            combined_summary=self._make_combined_summary(),
            classifiers=self.classifier_summaries,
            plot_data=self._make_plot_data(stim=False, classifier=True,
                                           joint=joint),
            sme_table=self._make_sme_table(),
            feature_data = self._make_feature_plots(),
            joint=joint
        )

    def generate_open_loop_fr_report(self):
        """ Generate an open-loop stim report

        Returns
        -------
        Rendered open loop report as a string.

        """
        return self._render(
            'FR2',
            stim=True,
            combined_summary=self._make_combined_summary(),
            stim_params=FRStimSessionSummary.stim_parameters(self.session_summaries),
            recall_tests=FRStimSessionSummary.recall_test_results(self.session_summaries, 'FR2'),
            hmm_results=self.hmm_results,
            plot_Data=self._make_plot_data(stim=True, classifier=False,
                                           biomarker_delta=False)

        )

    def generate_closed_loop_fr_report(self, experiment):
        """ Generate an FR5 report

        Returns
        -------
        Rendered FR5 report as a string.

        """
        return self._render(
            experiment,
            stim=True,
            combined_summary=self._make_combined_summary(),
            classifiers=self.classifier_summaries,
            stim_params=FRStimSessionSummary.stim_parameters(self.session_summaries),
            recall_tests=FRStimSessionSummary.recall_test_results(self.session_summaries, experiment),
            feature_data = self._make_feature_plots(),
            hmm_results=self.hmm_results,
            plot_data = self._make_plot_data(stim=True, classifier=True,
                                             biomarker_delta=True)
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

    def generate_ps5_report(self):
        """ Generate a PS5 report

        Returns
        -------
        Rendered PS5 report as a string
        """
        return self._render(
            'PS5',
            stim=True,
            combined_summary=self._make_combined_summary(),
            stim_params=FRStimSessionSummary.stim_parameters(self.session_summaries),
            recall_tests=FRStimSessionSummary.recall_test_results(self.session_summaries, 'PS5'),
            plot_data = self._make_plot_data(stim=True, classifier=False,
                                             biomarker_delta=True)
        )
