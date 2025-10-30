import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal

def test_get_strand_motifs_productions_trajectory():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    simulations_run_no=0
    motiflengths = [2,3,4,5]

    filepaths = kdi.utils._create_ligations_filepath_lists(strand_trajectory_id,param_file_no)[0]

    for motiflength in motiflengths:
        maximum_ligation_window_length = motiflength
        smpt = kdi.get.strand_motifs_productions_trajectory(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                simulations_run_no=0)

        assert(kdi.isinstance_motifproductiontrajectory(smpt))
        assert_equal(smpt.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(smpt.motiflength,motiflength)
        strand_motifs_productions_trajectory = kdi.read.strand_motifs_productions_trajectory(
                filepaths,
                smpt.alphabet,
                motiflength,
                maximum_ligation_window_length
                )
        assert_equal(smpt.times.val, strand_motifs_productions_trajectory.times.val)
        assert_equal(smpt.productions.val, strand_motifs_productions_trajectory.productions.val)
        assert(kdi.are_compatible_motif_production_trajectories(smpt,strand_motifs_productions_trajectory))
