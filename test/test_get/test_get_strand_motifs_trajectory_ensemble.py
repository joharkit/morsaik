import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_get_strand_motifs_trajectory_ensemble():
    """
    compares get_strand_motifs_trajectory_ensemble with
    read_strand_motifs_trajectory_ensemble
    """
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    motiflengths = [2,3,4,5]


    for motiflength in motiflengths:
        filepath_lists = kdi.utils._create_complexes_filepath_lists(strand_trajectory_id,param_file_no)
        rsmte = kdi.read.strand_motifs_trajectory_ensemble(filepath_lists,
                alphabet = ['A','T'],
                motiflength = motiflength)
        gsmte = kdi.get.strand_motifs_trajectory_ensemble(
                motiflength,
                strand_trajectory_id,
                param_file_no=0)

        assert(kdi.isinstance_motiftrajectoryensemble(gsmte))
        assert_equal(gsmte.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(gsmte.motiflength,motiflength)

        for trajectory_index in range(len(gsmte.trajectories)):
            gsmt = gsmte.trajectories[trajectory_index]
            rsmt = rsmte.trajectories[trajectory_index]
            assert_equal(gsmt.times.val, rsmt.times.val)
            assert(gsmt.times.domain[0].units==rsmt.times.domain[0].units)
            assert_equal(gsmt.motifs, rsmt.motifs)
            assert(kdi.are_compatible_motif_trajectories(gsmt,rsmt))
        assert(kdi.are_compatible_motif_trajectory_ensembles(gsmte,rsmte))
