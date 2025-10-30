import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_get_strand_motifs_productions_trajectory_ensemble():
    """
    compares get_strand_motifs_productions_trajectory_ensemble with
    read_strand_motifs_productions_trajectory_ensemble
    """
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
        gsmpte = kdi.get.strand_motifs_productions_trajectory_ensemble(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                maximum_ligation_window_length=maximum_ligation_window_length,
                )

        assert(kdi.isinstance_motifproductiontrajectoryensemble(gsmpte))
        assert_equal(gsmpte.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(gsmpte.motiflength,motiflength)
        assert_equal(gsmpte.maximum_ligation_window_length,maximum_ligation_window_length)
        assert_equal(gsmpte.number_of_letters,len(rsmpte.alphabet))
        assert_equal(gsmpte.unit,rsmpte.unit)

        for trajectory_index in range(len(gsmpte.trajectories)):
            gsmt = gsmpte.trajectories[trajectory_index]
            rsmt = rsmpte.trajectories[trajectory_index]
            assert_equal(gsmt.times.val, rsmt.times.val)
            assert_equal(gsmt.productions.val, rsmt.productions.val)
            assert(kdi.are_compatible_motif_production_trajectories(gsmt,rsmt))
        assert(kdi.are_compatible_motif_production_trajectory_ensembles(gsmpte,rsmpte))
