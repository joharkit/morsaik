import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest


def test_read_strand_length_distribution():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    simulations_run_no=0
    alphabet = kdi.get.alphabet(strand_trajectory_id)

    filepaths = []
    for simulations_no in [0,1]:
        filepaths += [f"./data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_{simulations_no}/length_distribution.txt"]

    strand_length_distribution = kdi.read.strand_length_distribution(filepaths, alphabet)
    assert strand_length_distribution['steps'][0]==1
    assert strand_length_distribution['times'][0]==0.00141473
    assert strand_length_distribution['mean_length'][0]==1.00806
    sld0 = strand_length_distribution['strand_length_distribution'][0]
    assert np.all(sld0[sld0>0.]==np.array([4920,40]))

    assert strand_length_distribution['steps'][-1]==5351479261
    assert strand_length_distribution['times'][-1]==1.15059e11
    assert strand_length_distribution['mean_length'][-1]==2500
    desired = np.zeros(3117)
    desired[1882] = 1
    desired[3116] = 1
    assert np.all(strand_length_distribution['strand_length_distribution'][-1]==desired)
