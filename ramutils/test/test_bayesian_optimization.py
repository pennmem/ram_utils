import pytest
import functools
import numpy as np

from pkg_resources import resource_filename
from ramutils.bayesian_optimization import choose_location

datafile = functools.partial(resource_filename,
                             'ramutils.test.test_data.input')


def test_choose_location():
    dataset_loc_0 = np.load(
        datafile('/bayesian_optimization/dataset_loc_0.npy'))
    dataset_loc_1 = np.load(
        datafile('/bayesian_optimization/dataset_loc_1.npy'))
    loc_name_0 = "LA7_LA8"
    loc_name_1 = "LC6_LC7"
    bounds = np.load(datafile('/bayesian_optimization/bounds.npy'))
    decision, loc_info = choose_location(dataset_loc_0, loc_name_0,
                                         dataset_loc_1, loc_name_1, bounds,
                                         alpha=1e-3, epsilon=1e-3,
                                         sig_level=0.05)

    assert np.isclose(decision['p_val'], 0.006084130127878487, atol=1e-4)
    assert np.isclose(decision['t_stat'], 2.50722681211564, atol=1e-4)
    assert decision['Tie'] == 0
    assert decision['best_location_name'] == 'LA7_LA8'

    return
