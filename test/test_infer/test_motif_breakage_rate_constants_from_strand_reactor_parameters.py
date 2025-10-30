import morsaik as kdi

def test_infer_motif_breakage_rate_constants_from_strand_reactor_parameters():
    strand_trajectory_id = '9999_99_99__99_99_99'
    motiflength = 4
    complements = [1,0]
    strand_reactor_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    alphabet = kdi.get.alphabet(strand_trajectory_id)

    kdi.infer.motif_breakage_rate_constants_from_strand_reactor_parameters(
        strand_reactor_parameters,
        motiflength,
        alphabet,
        complements
    )
