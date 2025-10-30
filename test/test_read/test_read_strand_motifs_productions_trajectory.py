import morsaik as kdi
from numpy.testing import assert_equal
import pytest

def test_read_strand_motifs_productions():
    filepaths = ['data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_1/ligation_statistics.txt',]
    alphabet = ['A','T']
    motiflengths = [2,3,4,5,6]

    for motiflength in motiflengths:
        maximum_ligation_window_length = motiflength
        kdi.read.strand_motifs_productions_trajectory(filepaths,
                alphabet,
                motiflength,
                maximum_ligation_window_length)

def test__step_simulation_and_ligation_strands_array():
    filepath = 'data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_1/ligation_statistics.txt'

    desired = ('AAATAATTAATTAAATTAATTATAAATTAATTTAATTTATATTTATTTTTATAAATTAAATATTATTAATTAAATATTATTATTTATTTATTAATAAAAAATAATTATTATTTAATTATATAATATAATAATTTTATATTATTAATTAAAATATATTTATT',
            'ATATAAATAATAATAATATTATTATTTAAATTTTATTTATAAATAAAATTTAAATATAATT',
            'ATTAATATTATTATATAATTATTTTTAT',
            'TATATATATTTTTAAAATTATTTAATAATTATAAAT')
    _, _, ls = kdi.read._step_simulation_and_ligation_strands_array(filepath) 

    assert_equal(ls[0][0],desired)

def test_read__motif_production_reactants():
    motiflength=5
    desired = ('TATT', 'ATAT', 'ATTAT', 'continuation_5_2_continuation')
    filepath = 'data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_1/ligation_statistics.txt'
    _, _, ls = kdi.read._step_simulation_and_ligation_strands_array(filepath)
    print(ls[0][0])
    assert_equal(
            set(kdi.read._motif_production_reactants(*ls[0][0],motiflength,motiflength)),
            set(desired)
            )

    _,_,ligation_reactants = kdi.read._step_simulation_and_ligation_strands_array(filepath)
    assert(ligation_reactants[0][370]==('AT', 'AAATTTAAATTA', 'TTT', 'ATAAA'))

def test_read__ligation_statistics():
    motiflength=5

    filepath = 'data/9999_99_99__99_99_99/data_exp/data_param_file_0/data_simulations_run_0/data_simulations_1/ligation_statistics.txt'
    ending_reactant_sequence = 'TAATAATTTTATATTATTAATTAAAATATATTTATT'
    leaving_reactant_sequence = 'ATATAAATAATAATAATATTATTATTTA'
    template_second_part = 'TAAATATTAATAATTTATTAAAATTTTTATATATAT'[::-1]
    template_first_part = 'TATTTTTATTAATATATTATTATAATTA'[::-1]

    lsttstcs = kdi.read._ligation_statistics(filepath)
    assert(lsttstcs[2][0][0][-3]==ending_reactant_sequence)
    assert(lsttstcs[2][0][0][-2]==leaving_reactant_sequence)

    ending_motif = ending_reactant_sequence[-(motiflength-1):]
    leaving_motif = leaving_reactant_sequence[:(motiflength-1)]
    product_motif = ending_motif[-(motiflength-motiflength//2):]+leaving_motif[:motiflength//2]
    template_motif = template_first_part[-(motiflength//2):]+template_second_part[:(motiflength-motiflength//2)]
    production_channel_id = kdi.read._determine_production_channel_id(
            ending_motif,
            leaving_motif,
            template_first_part,
            template_second_part,
            motiflength,
            motiflength-motiflength//2-1, #ligation_spot
            motiflength
            )
    assert_equal(
            kdi.read._motif_production_reactants(ending_reactant_sequence,
            leaving_reactant_sequence, template_first_part,
            template_second_part,motiflength, motiflength),
            (product_motif, ending_motif, leaving_motif, template_motif,
                production_channel_id)
            )

    _,_, ligation_statistics = kdi.read._ligation_statistics(fname=filepath, skiprows=2)
    desired_ligation_statistics_0_370 = (2, 3, 2, 12, 8, 0, 0, 0, 0, 0.008208, 0.00392,
            '[["5TTTAT3",0],["-","X"],["5AAAATTAA3","3ATTAAATT5"],["X","-"],[0,"3T5"],["X","-"],["5TTT3","3AAA5"],["-","|"],["5AT3","3TA5"],["-","X"],["5AAA3",0]]',
            'AT', 'AAA', 1)
    assert(ligation_statistics[0][370]==desired_ligation_statistics_0_370)
