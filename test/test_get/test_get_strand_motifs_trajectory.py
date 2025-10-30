import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal

def test_get_strand_motifs_trajectory():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    simulations_run_no=0
    motiflengths = [2,3,4,5]

    filepaths = kdi.utils._create_complexes_filepath_lists(strand_trajectory_id,param_file_no)[0]

    for motiflength in motiflengths:
        smt = kdi.get.strand_motifs_trajectory(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                simulations_run_no=0)

        assert(kdi.isinstance_motiftrajectory(smt))
        assert_equal(smt.alphabet,
                kdi.get.alphabet(strand_trajectory_id))
        assert_equal(smt.motiflength,motiflength)
        strand_motif_trajectory = kdi.read.strand_motifs_trajectory(filepaths,smt.alphabet,motiflength)
        assert_equal(smt.times.val, strand_motif_trajectory.times.val)
        assert_equal(smt.times.domain[0].units, strand_motif_trajectory.times.domain[0].units)
        assert_equal(smt.motifs, strand_motif_trajectory.motifs)
        assert(kdi.are_compatible_motif_trajectories(smt,strand_motif_trajectory))
