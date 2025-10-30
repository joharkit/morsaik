import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_read_strand_motifs_productions_trajectory_ensemble():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    motiflengths = [2,3,4,5]


    for motiflength in motiflengths:
        maximum_ligation_window_length = motiflength
        filepath_lists = kdi.utils._create_ligations_filepath_lists(strand_trajectory_id,param_file_no)
        rsmpte = kdi.read.strand_motifs_productions_trajectory_ensemble(filepath_lists,
                ['A','T'],
                motiflength,
                maximum_ligation_window_length)

        assert(kdi.isinstance_motifproductiontrajectoryensemble(rsmpte))
        assert_equal(rsmpte.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(rsmpte.motiflength,motiflength)

        for trajectory_index in range(len(rsmpte.trajectories)):
            rsmt = rsmpte.trajectories[trajectory_index]
            assert(kdi.isinstance_motifproductiontrajectory(rsmt))
