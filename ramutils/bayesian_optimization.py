import collections

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm,sem
from sklearn.gaussian_process.kernels import Matern, WhiteKernel


def choose_location(dataset_loc_0, loc_name_0, dataset_loc_1, loc_name_1,
                    bounds, sham_delta_classifier_vector=None,
                    gp_params=None, alpha=1e-3, epsilon=1e-3, sig_level=0.05):
    """ Given the result of a parameter search session, return a decision dict
        decision = 1, loc1 > loc2
                = 0 , inconclusive
                = -1, loc1 < loc2
        Tie = 1 if two-sample t test insignificant
            = 0 of two-sample t test significant

    Parameters:
    -----------
    dataset_loc1: PS dataset for loc1, Nx2 numpy array
    dataset_loc2: PS dataset for loc2, Nx2 numpy array
    gp_params: params for gaussian process regressor (set to sklearn defaults)
    alpha:
    epsilon:
    sig_level: significant level for t-test

    Returns
    -------
    decision
    loc_info
    """
    # set up kernels
    kernel_matern = Matern() + WhiteKernel(noise_level=1)
    model = gp.GaussianProcessRegressor(kernel=kernel_matern, alpha=alpha,
                                        n_restarts_optimizer=5,
                                        normalize_y=True)

    if dataset_loc_1 is None or not loc_name_1:
        dataset_loc_1 = np.zeros((0,2))
        loc_name_1 = ''

    xp_loc = [np.array(map(lambda x: [x], dataset_loc_0[:, 0])),
              np.array(map(lambda x: [x], dataset_loc_1[:, 0]))]
    yp_loc = [np.array(map(lambda x: [x], dataset_loc_0[:, 1])),
              np.array(map(lambda x: [x], dataset_loc_1[:, 1]))]

    x_max_loc = [None] * 2
    y_max_loc = [None] * 2
    sem_list = [None] * 2
    SNR = [None] * 2

    loc_info = collections.OrderedDict()
    loc_names = [loc_name_0, loc_name_1]

    for i, loc_name in enumerate(loc_names):
        x_max_loc[i], y_max_loc[i], sem_list[i], SNR[i] = find_max(xp_loc[i],
                                                                   yp_loc[i],
                                                                   model,
                                                                   np.reshape(bounds[i, :], (1, 2)))
        loc_info[loc_name] = collections.OrderedDict()
        loc_info[loc_name]['amplitude'] = x_max_loc[i].flatten()[0]
        loc_info[loc_name]['delta_classifier'] = y_max_loc[i].flatten()[0]
        loc_info[loc_name]['sem'] = sem_list[i].flatten()[0]
        loc_info[loc_name]['snr'] = SNR[i]
        loc_info[loc_name]['loc_name'] = loc_name

    t_stat = (y_max_loc[0] - y_max_loc[1]) / np.sqrt((sem_list[0]) ** 2 + (sem_list[1]) ** 2)
    p_val = norm.cdf(- np.abs(t_stat))

    decision = collections.OrderedDict()  # store decision

    decision['p_val'] = p_val.flatten()[0]
    decision['t_stat'] = t_stat.flatten()[0]

    if p_val < sig_level:
        decision['Tie'] = 0
        if t_stat > 0:
            decision['best_location_name'] = loc_names[0]
        else:
            decision['best_location_name'] = loc_names[1]
    else:
        decision['Tie'] = 1
        if sem_list[0] < sem_list[1]:
            decision['best_location_name'] = loc_names[0]
        else:
            decision['best_location_name'] = loc_names[1]

    # comparing "the champion" to  sham
    champion_mean = loc_info[decision['best_location_name']]['delta_classifier']
    champion_sem = loc_info[decision['best_location_name']]['sem']
    if sham_delta_classifier_vector is not None:
        sham_mean = np.mean(sham_delta_classifier_vector)
        sham_sem = sem(sham_delta_classifier_vector)
        t_stat_champ_sham = (champion_mean - sham_mean) / np.sqrt((champion_sem) ** 2 + (sham_sem) ** 2)
        p_val_champ_sham = norm.cdf(- np.abs(t_stat_champ_sham))
        decision['p_val_champ_sham'] = p_val_champ_sham
        decision['t_stat_champ_sham'] = t_stat_champ_sham
        decision['sham_delta_classifier'] = sham_mean
        decision['sham_sem'] = sham_sem
    else:
        decision['p_val_champ_sham'] = decision['t_stat_champ_sham'] =decision['sham_delta_classifier'] = decision['sham_sem'] = 'N/A'
    return decision, loc_info


def find_max(xp, yp, model, bounds, n_samp=100):
    model.fit(xp, yp)
    evaluated_loss = model.predict(xp)
    opt_loc = np.argmax(evaluated_loss)
    x_max = xp[opt_loc]
    y_max = model.predict(x_max.reshape(1,1))

    kernel = model.kernel_.k1
    sigma_noise = model.kernel_.k2.noise_level
    x_joint = np.concatenate((xp, np.array([x_max]).reshape(1, 1)))

    # calculate standard error of predicted mean
    T = xp.shape[0]
    sigma_joint = kernel(x_joint)
    K = sigma_joint[:-1, :-1] + sigma_noise * np.eye(T, T)
    k = sigma_joint[:-1, -1]
    kp1 = sigma_joint[-1, -1] + sigma_noise
    mu_pred = np.dot(np.dot(k.T, np.linalg.inv(K + sigma_noise * np.eye(T, T))), yp)
    sigma_pred = kp1 - np.dot(np.dot(k.T, np.linalg.inv(K + sigma_noise * np.eye(T, T))), k)
    sigma_cond = np.dot(k.T, np.linalg.inv(K + sigma_noise * np.eye(T, T)))

    se = np.sqrt(np.dot(sigma_cond, sigma_cond.T) * sigma_noise)
    SNR = np.mean(np.abs(evaluated_loss)) / np.sqrt(sigma_noise)  # Signal to noise ratio

    return x_max, y_max, se, SNR


def target(x):
    return np.exp(-(x - 0.8) ** 2) + np.exp(-(x - 0.2) ** 2) - 0.5 * (x - 0.8) ** 3 - 2.0

