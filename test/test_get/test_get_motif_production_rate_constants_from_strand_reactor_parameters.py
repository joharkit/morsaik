import morsaik as kdi

def test_get_motif_production_rate_constants_from_strand_reactor_parameters():
    strand_trajectory_id = '9999_99_99__99_99_99'
    motiflength = 4
    complements = [1,0]
    motif_production_rate_constants = kdi.get.motif_production_rate_constants_from_strand_reactor_parameters(
        strand_trajectory_id,
        motiflength,
        complements,
        )
    keys = {'productions', 'motiflength', 'alphabet', 'number_of_letters', 'unit', 'maximum_ligation_window_length'}
    assert(keys==set(motif_production_rate_constants._asdict()))
