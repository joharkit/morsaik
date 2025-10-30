import morsaik as kdi
import numpy as np

def test_get_strand_reactor_parameters():
    strand_reactor_id = '9999_99_99__99_99_99'
    param_file_no = 0
    srp = kdi.get.strand_reactor_parameters(strand_reactor_id,
            param_file_no=param_file_no)
    filepath = kdi.utils._create_typical_strand_parameters_filepath(strand_reactor_id,
            param_file_no=param_file_no)
    test_srp = kdi.read.strand_reactor_parameters(filepath)
    assert(srp == test_srp)
