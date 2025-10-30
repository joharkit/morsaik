import pytest
from numpy.testing import assert_equal
import morsaik as kdi

@pytest.mark.xfail
def test_motif_production_rate_constants_from_strand_reactor_parameters():
    motiflengths = [4,]# [2,3,5,6,10]
    alphabet = ['A','T']
    strand_reactor_id = '9999_99_99__99_99_99'
    param_file_no = 0
    desired_strand_reactor_parameters_filepath = 'data/'+strand_reactor_id+'/all_parameter_files/parameters_{}.txt'.format(param_file_no)
    actual_strand_reactor_parameters_filepath = kdi.utils._create_typical_strand_parameters_filepath(
        strand_reactor_id,
        param_file_no=param_file_no,
    )
    complements = [1,0]

    strand_reactor_parameters = kdi.read.strand_reactor_parameters(
        actual_strand_reactor_parameters_filepath,
    )
    for motiflength in motiflengths:
        maximum_ligation_window_length = motiflength
        actual_mprc = kdi.infer.motif_production_rate_constants_from_strand_reactor_parameters(
            strand_reactor_parameters,
            motiflength,
            alphabet,
            maximum_ligation_window_length,
            complements,
        )

        assert False
