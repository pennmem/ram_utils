""" Set of utility functions for generating matplotlib plots for RAM reports """
import os
import matplotlib
import numpy as np

# No graphical display. Must be called before importing pyplot
matplotlib.use('Agg')

import matplotlib.pyplot as plt

LINE_WIDTH = 2


def roc_curve(fpr, tpr):
    """ Plot ROC Curve given false positive rate and true positive rate
    
    Parameters:
    -----------
    fpr: array_like
        List of false positive rates
    tpr: array_like
        List of true positive rates

    Returns
    -------
    ax
        Matplotlib axes

    """
    _, ax = plt.subplots()

    ax.plot(fpr,
            tpr,
            linestyle='solid',
            linewidth=LINE_WIDTH)

    # Plotting y=x for reference
    x = np.arange(0, 1.1, .1)
    y = x
    ax.plot(x,
            y,
            linestyle='dotted',
            linewidth=1)

    ax.set(xlabel='False Alarm Rate',
           ylabel='Hit Rate',
           xlim=(0, 1),
           ylim=(0, 1))

    return ax


def repeitition_ratio(irt_within_cat, irt_between_cat):
    """ Barplot of inter-response times for transitions within and between
        categories

    Parameters:
    -----------
    irt_within_cat: int
        Inter-response time for within category items
    irt_between_cat: int
        Inter-response times for between category items
    
    Returns:
    --------
    ax
        Matplotlib axes
    
    """
    _, ax = plt.subplots()
    ax.bar([0, 1], [irt_within_cat, irt_between_cat], color='0.5')
    ax.set(ylabel="IRT (msec)",
           xticks=[0, 1],
           xticklabels=['Within Cat', 'Between Cat'])

    return ax


def tercile_classifier_estimate(percent_diff_from_mean):
    """ Plot the percent change in recall at the terciles of classifier output

    Parameters
    ----------
    percent_diff_from_mean: array_like (shape=(3,))
        Array of differences from the mean
    
    Returns:
    --------
    ax
        Matplotlib axes
    
    """
    if len(percent_diff_from_mean) != 3:
        raise RuntimeError("Input should be an array of length 3")

    _, ax = plt.subplots()

    ax.bar([0, 1, 2], percent_diff_from_mean, color='0.5')
    ax.axhline(0, linestyle='dashed', color='0.5')

    ax.set(ylabel="Recall Change From Mean (%)",
           xticks=[0, 1, 2],
           xticklabels=['Low', 'Middle', 'High'])

    return ax


def probability_lineplot(serial_positions, prob_of_recall,
                         xlabel="Serial Position",
                         ylabel="Probability of Recall", labels=[]):
    """ Plot probability of recall as a function of serial position

    Parameters
    ----------
    serial_positions: list
        Lists of serial positions, usually 1-12 inclusive (x-axis)
    prob_of_recall: list
        List of recall probabilities. This could be overall probability of
        recall or probability of first recall. List of lists is also accepted
        if multiple lineplots are desired
    labels: list
        List of labels to use for legend if multiple lines are plotted

    Returns
    -------
    ax
        Matplotlib axes object containing the plot

    """
    is_array = (type(prob_of_recall) == list)
    print(is_array)
    if (is_array) and (len(labels) != len(prob_of_recall)):
        raise RuntimeError(
            "Number of labels should match the number of recall prob arrays")

    _, ax = plt.subplots(1)

    # Handle single-line case
    if is_array is False:
        ax.plot(serial_positions,
                prob_of_recall,
                linestyle='solid',
                marker='o',
                linewidth=LINE_WIDTH)

    # Multi-line case
    else:
        for i in range(len(prob_of_recall)):
            ax.plot(serial_positions,
                    prob_of_recall[i],
                    linestyle='solid',
                    marker='o',
                    linewidth=LINE_WIDTH,
                    label=labels[i])

    ax.set(xlabel="Serial Position",
           ylabel="Probability of Recall",
           ylim=(0, 1),
           xticks=serial_positions)

    return ax


def stim_recall():
    return


def probability_of_stimulation():
    return


def delta_recall():
    return


def biomarker_distribution():
    return


def post_stim_eeg():
    return
