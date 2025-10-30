import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_read_strand_motifs_trajectory_ensemble():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    motiflengths = [2,3,4,5]


    for motiflength in motiflengths:
        filepath_lists = kdi.utils._create_complexes_filepath_lists(strand_trajectory_id,param_file_no)
        rsmte = kdi.read.strand_motifs_trajectory_ensemble(filepath_lists,
                alphabet = ['A','T'],
                motiflength = motiflength)

        assert(kdi.isinstance_motiftrajectoryensemble(rsmte))
        assert_equal(rsmte.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(rsmte.motiflength,motiflength)

        for trajectory_index in range(len(rsmte.trajectories)):
            rsmt = rsmte.trajectories[trajectory_index]
            assert(kdi.isinstance_motiftrajectory(rsmt))
