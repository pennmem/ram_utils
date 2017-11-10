import pymc3 as pm
import numpy as np
import pandas as pd


class HierarchicalModel(object):
    """ Class for running pymc3 models based on the experiment type and subject """

    def __init__(self, data, subject, experiment, item_comparison='list'):
        """

        Parameters
        ----------
        data: pd.DataFrame
            Pandas DataFrame containing the data to be used for model fitting.
            Typically, this should be the fr-stim-table produced in the automated
            reports
        subject: str
            Subject identifier
        experiment: str
            Task name. Used to determine which model to use with fit() is called
        item_comparison: str, default 'list'
            Controls the type of comparison used in the model. The default is 'list'
            indicating stim list vs. non-stim list. Other options are 'stim' and
            'post_stim' indicating that stim items vs. low-biomarker non stim items
            should be compared and post stim items vs. low biomarker non stim items,
            respectively.

        Notes
        -----
        Currently models are identical for each experiment type. Ideally, the model
        will be experiment-specific in the future, so a factory method fit() is
        used to dispatch the appropriate private method based on the experiment
        type.

        """
        self.subject = subject
        self.experiment = experiment
        self.item_comparison = item_comparison

        self.data = self._prepare_data(data)  # data prep needs to happen before other object attributes are set
        self.n_sessions = len(self.data.session.unique())
        self.n_serialpos = len(self.data.serialpos.unique())
        self.n_lists = len(self.data.list.unique())
        self.session_idx = self.data.session.values
        self.list_idx = self.data.list.values
        self.serialpos_idx = self.data.serialpos.values

        level_treatment_map = {
            'list' : self.data.is_stim_list.values,
            'stim' : self.data.is_stim_item.values,
            'post_stim': self.data.is_post_stim_item
        }
        self.treatment_vals = level_treatment_map[self.item_comparison]
        self.model = self._build_baseline_model()
        self.trace = None
        return

    def fit(self, draws=5000, tune=1000):
        """ Fit a hierarchical model for the given subject and experiment

        Returns
        -------
        trace (pymc3.backends.base.MultiTrace)
            A MultiTrace object that contains the samples

        """
        method_str = "_fit_{}_model".format(self.experiment)
        dispatch_method = getattr(self, method_str)
        return dispatch_method(draws, tune)

    def _prepare_data(self, raw_data):
        """ Set the data attribute for the model after some cleaning based on the
            requested model level from initialization
        """
        df = raw_data[raw_data.list > 3]

        # serialpos, list, session all need to be 0-indexed for the vectorized
        # implementation of the model to work
        df["session"] = df["session"] - (df["session"].min())
        df["serialpos"] = df["serialpos"] - (df["serialpos"].min())

        # Turn list into a % session completed variable to that subjects who complete only partial
        # sessions can still be compared to full sessions
        df["list"] = (df["list"] - (df["list"].min())) / (len(df["list"].unique()))

        serialpos_dummies = pd.get_dummies(df.serialpos)
        df = pd.concat([df, serialpos_dummies], axis=1)

        df["low_biomarker"] = (df["prob"] < df["thresh"])

        stim_or_low_bio_df = df[((df["is_stim_item"] == True) |
                                 ((df["low_biomarker"] == True) &
                                 (df["is_stim_item"] == False)))]

        post_stim_or_low_bio_df  = df[((df["is_post_stim_item"] == True) |
                                       ((df["low_biomarker"] == True) &
                                       (df["is_post_stim_item"] == False)))]

        level_data_map = {"list": df,
                          "stim": stim_or_low_bio_df,
                          "post_stim": post_stim_or_low_bio_df}
        data = level_data_map[self.item_comparison]
        return data

    def _build_baseline_model(self):
        with pm.Model() as model:
            # Treat list as a continuous variable
            listpos_coef = pm.Normal('listnum', mu=0, sd=1)

            # Treat serial position as a factor variable
            serialpos_coef = pm.Normal('serialpos', mu=0, sd=1, shape=11)

            # Random intercept hyperpriors
            mu_alpha = pm.Normal('avg_recall', mu=0, sd=1)
            sigma_alpha = pm.HalfNormal('std_recall', sd=1)

            # Random slope hyperpriors
            mu_beta = pm.Normal('Stim Effect (Across Sessions)', mu=0., sd=1)
            sigma_beta = pm.HalfNormal('std_stim_effect', sd=1)

            # Model the recall probability offset by session (session specific intercept)
            alpha_offset = pm.Normal("session_avg_recall_offset", mu=0, sd=1, shape=self.n_sessions)
            alpha = pm.Deterministic("session_avg_recall", (mu_alpha + alpha_offset * sigma_alpha))

            # Model the stim effect offset by session (session specific slope)
            beta_offset = pm.Normal("session_stim_effect_offset", mu=0, sd=1, shape=self.n_sessions)
            beta = pm.Deterministic("Stim Effect (Session Level)", mu_beta + beta_offset * sigma_beta)

            # Expected value
            recall_est = pm.invlogit(alpha[self.data.session.values] +
                                     serialpos_coef[0] * self.data[0] +
                                     serialpos_coef[1] * self.data[1] +
                                     serialpos_coef[2] * self.data[2] +
                                     serialpos_coef[3] * self.data[3] +
                                     serialpos_coef[4] * self.data[4] +
                                     serialpos_coef[5] * self.data[6] +
                                     serialpos_coef[6] * self.data[7] +
                                     serialpos_coef[7] * self.data[8] +
                                     serialpos_coef[8] * self.data[9] +
                                     serialpos_coef[9] * self.data[10] +
                                     serialpos_coef[10] * self.data[11] +
                                     listpos_coef * self.data.list.values +
                                     beta[self.data.session.values] * self.data.is_stim_list)

            # single_subject_data likelihood
            y_like = pm.Bernoulli('y_like', recall_est, observed=self.data.recalled)

        return model

    def _fit_FR3_model(self, draws, tune):
        with self.model:
            self.trace = pm.sample(draws=draws, tune=tune)[tune:]
        return self.trace

    def _fit_FR5_model(self, draws, tune):
        return self._fit_FR3_model(draws, tune)

    def _fit_catFR3_model(self, draws, tune):
        return self._fit_FR3_model(draws, tune)

    def _fit_catFR5_model(self, draws, tune):
        return self._fit_FR3_model(draws, tune)


class HierarchicalModelPlots:
    """ Custom plots for hierarchical models to estimate the effect of stim """

    def __init__(self, trace):
        self.trace  = trace
        self.stimvars = ["Stim Effect (Across Sessions)", "Stim Effect (Session Level)"]

    def traceplot(self):
        """ Generate a custom traceplot based on the trace from the fit model """

        line_dict = dict(zip(self.stimvars, [0]*len(self.stimvars)))
        ax = pm.traceplot(self.trace, lines=line_dict)
        return ax

    def forestplot(self, title):
        """ Generate a custom forestplot using the trace from the fit model """
        session_values = self.trace.get_values("Stim Effect (Session Level)", chains=[self.trace.chains[0]][0])
        num_sessions = np.shape(session_values)[1]
        session_titles = [" ".join(["Session", str(i)]) for i in range(num_sessions)]
        ylabels = ["Agg"] + session_titles

        ax = pm.forestplot(self.trace,
                           varnames=self.stimvars,
                           main=title,
                           xtitle="Estimated Coefficient for the Effect of Stimulation",
                           ylabels = ylabels,
                           quartiles=False,
                           linewidth=5,
                           color='#136ba5',
                           markersize=6,
                           fontsize=12)

        return ax

