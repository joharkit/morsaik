import morsaik as kdi
from numpy.testing import assert_equal

def test_get_motif_breakage_rate_constants_from_strand_reactor_parameters():
    motiflengths = [4]
    complements = [1,0]
    strand_trajectory_id = '9999_99_99__99_99_99'
    for motiflength in motiflengths:
        strand_reactor_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
        alphabet = kdi.get.alphabet(strand_trajectory_id)
        actual = kdi.get.motif_breakage_rate_constants_from_strand_reactor_parameters(
            strand_trajectory_id,
            motiflength,
            complements,
        )

        desired = kdi.infer.motif_breakage_rate_constants_from_strand_reactor_parameters(
            strand_reactor_parameters,
            motiflength,
            alphabet,
            complements
        )
        assert_equal(actual.motiflength, desired.motiflength)
        assert_equal(actual.alphabet, desired.alphabet)
        assert_equal(actual.unit, desired.unit)
        assert_equal(actual.number_of_letters, desired.number_of_letters)
        for key in desired.breakages.keys():
            assert_equal(actual.breakages[key].val, desired.breakages[key].val)
