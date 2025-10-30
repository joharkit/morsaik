import morsaik as kdi

def test_get_alphabet():
    strand_trajectory_id = '9999_99_99__99_99_99'
    assert(kdi.get.alphabet(strand_trajectory_id)==['A','T'])
