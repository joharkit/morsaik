import morsaik as kdi
import numpy as np

def test_read_strand_reactor_parameters():
    strand_reactor_id = '9999_99_99__99_99_99'
    param_file_no = 0
    parameters_to_read_from_files = [None,['c_ref']]
    for parameters_to_read_from_file in parameters_to_read_from_files:
        filepath = kdi.utils._create_typical_strand_parameters_filepath(strand_reactor_id,
                param_file_no=param_file_no)
        srp = kdi.read.strand_reactor_parameters(filepath,
                parameters_to_read_from_parameters_file=parameters_to_read_from_file)
        # test output format
        assert(isinstance(srp,dict))
        if parameters_to_read_from_file is None:
            ptrff = kdi.utils._return_parameters_to_read_from_parameters_file()
        else:
            ptrff = parameters_to_read_from_file
        # test all parameters keys are in dictionary
        barray = [pp in srp.keys() for pp in ptrff]
        assert(np.prod(barray))
