import morsaik as kdi

def test__return_parameters_to_read_from_parameters_file():
    parameters_list = kdi.utils._return_parameters_to_read_from_parameters_file()
    assert(isinstance(parameters_list,list))

def test__create_typical_strand_reactor_dirpath():
    strand_trajectory_id = '9999_99_99'
    dirpath = kdi.utils._create_typical_strand_reactor_dirpath(strand_trajectory_id)
    test_dirpath = './data/'
    test_dirpath += strand_trajectory_id + '/'
    assert(dirpath==test_dirpath)

def test__create_typical_strand_trajectory_ensemble_dirpath():
    strand_trajectory_id = '9999_99_99'
    param_file_no = 2
    dirpath = kdi.utils._create_typical_strand_trajectory_ensemble_dirpath(strand_trajectory_id,
            param_file_no=param_file_no)
    test_dirpath = kdi.utils._create_typical_strand_reactor_dirpath(strand_trajectory_id)
    test_dirpath += 'data_exp/data_param_file_{}/'.format(param_file_no)
    assert(dirpath==test_dirpath)

def test__create_typical_strand_trajectory_dirpath():
    strand_trajectory_id = '9999_99_99'
    param_file_no = 2
    simulations_run_no = 39
    dirpath = kdi.utils._create_typical_strand_trajectory_dirpath(
            strand_trajectory_id,
            param_file_no=param_file_no,
            simulations_run_no=simulations_run_no
            )
    test_dirpath = kdi.utils._create_typical_strand_trajectory_ensemble_dirpath(
            strand_trajectory_id,
            param_file_no)
    test_dirpath += 'data_simulations_run_{}/'.format(simulations_run_no)
    assert(dirpath==test_dirpath)

def test__create_typical_strand_trajectory_section_dirpath():
    strand_trajectory_id = '9999_99_99'
    param_file_no = 2
    simulations_run_no = 39
    simulations_no = 1294
    dirpath = kdi.utils._create_typical_strand_trajectory_section_dirpath(
            strand_trajectory_id,
            param_file_no=param_file_no,
            simulations_run_no=simulations_run_no,
            simulations_no=simulations_no
            )
    test_dirpath = kdi.utils._create_typical_strand_trajectory_dirpath(strand_trajectory_id,
            param_file_no=param_file_no,
            simulations_run_no=simulations_run_no)
    test_dirpath += 'data_simulations_{}/'.format(simulations_no)
    assert(dirpath==test_dirpath)

def test__create_typical_strand_parameters_filepath():
    strand_trajectory_id = '9999_99_99'
    param_file_no = 2
    filepath = kdi.utils._create_typical_strand_parameters_filepath(
            strand_trajectory_id,
            param_file_no=param_file_no
            )
    test_filepath = kdi.utils._create_typical_strand_reactor_dirpath(strand_trajectory_id)
    test_filepath += 'all_parameter_files/parameters_{}.txt'.format(param_file_no)
    assert(filepath==test_filepath)
