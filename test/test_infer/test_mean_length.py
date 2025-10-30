import morsaik as kdi

def test_infer_mean_length():
    strand_trajectory_id='9999_99_99__99_99_99'
    motiflengths = [2,3,4,5]

    for motiflength in motiflengths:
        smt = kdi.get.strand_motifs_trajectory(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                simulations_run_no=0)
        mean_length = kdi.infer.mean_length(smt)
        assert mean_length[0] == 1.+40./4960.
