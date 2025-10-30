import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest


def test_get_strand_length_distribution():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    simulations_run_no=0
    alphabet = kdi.get.alphabet(strand_trajectory_id)

    filepaths = []
    for simulations_no in [0,1]:
        filepaths += [f"./data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_{simulations_no}/length_distribution.txt"]

    for key in ['steps','times','mean_length','strand_length_distribution']:
        assert np.all(kdi.read.strand_length_distribution(filepaths, alphabet)[key] == kdi.get.strand_length_distribution(strand_trajectory_id)[0][key])
