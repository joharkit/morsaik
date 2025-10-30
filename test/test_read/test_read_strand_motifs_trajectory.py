import morsaik as kdi
import numpy as np
from numpy.testing import assert_equal
import pytest

def test_read_strand_motifs_trajectory():
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0
    simulations_run_no=0
    motiflengths = [2,3,4,5]
    alphabet = kdi.get.alphabet(strand_trajectory_id)

    filepath = kdi.utils._create_typical_strand_trajectory_section_complexes_filepath(strand_trajectory_id,param_file_no,simulations_run_no)
    filepaths = [filepath,]

    for motiflength in motiflengths:
        strand_motif_trajectory = kdi.read.strand_motifs_trajectory(filepaths,alphabet,motiflength)
        assert(kdi.isinstance_motiftrajectory(strand_motif_trajectory))
        emvd = kdi._create_empty_motif_vector_dct(motiflength,strand_motif_trajectory.alphabet)
        mvc = kdi.MotifVector(motiflength,strand_motif_trajectory.alphabet,'1')
        emv = mvc(emvd)
        emt = kdi.MotifTrajectory([emv,]*strand_motif_trajectory.times.shape[0],strand_motif_trajectory.times)
        assert(kdi.are_compatible_motif_trajectories(emt,strand_motif_trajectory))

def test_steps_and_times_and_complexes_from_complexes_txt():
    complexes_m1_m4 = [2, [['5T3', '3T5'], ['-', 'X'], ['5A3', 'none']]]
    trajectory_id = '9999_99_99__99_99_99'

    filepath = kdi.utils._create_typical_strand_trajectory_section_complexes_filepath(
            trajectory_id,
            param_file_no = 0,
            simulations_run_no = 0,
            simulations_no = 0,
            )

    steps, times, complexes = kdi.read.steps_and_times_and_complexes_from_complexes_txt(filepath)
    assert(times[-1]==25000.7)
    assert(steps[-1]==387942)
    assert_equal(complexes[-1][-4],complexes_m1_m4)

def test_steps_and_times_and_sequence_trajectory_from_complexes_txt():
    desired_steps = np.array([1,  38775,  77032, 115995, 154972, 194014, 232584, 271537, 310841, 349401, 387942])
    desired_times = np.array([1.41473e-03, 2.50007e+03, 5.00018e+03, 7.50025e+03, 1.00003e+04, 1.25003e+04, 1.50004e+04, 1.75004e+04, 2.00005e+04, 2.25006e+04, 2.50007e+04])
    desired_sequence_number_trajectory = [{'A': 2460, 'T': 2460, 'AT': 10, 'AA': 10, 'TA': 10, 'TT': 10},
      {'A': 2460, 'T': 2460, 'TA': 10, 'TT': 10, 'AA': 10, 'AT': 10},
      {'A': 2460, 'T': 2460, 'AA': 10, 'TT': 10, 'AT': 10, 'TA': 10},
      {'A': 2460, 'T': 2460, 'TA': 10, 'TT': 10, 'AT': 10, 'AA': 10},
      {'A': 2460, 'T': 2460, 'AT': 10, 'TA': 10, 'TT': 10, 'AA': 10},
      {'A': 2460, 'T': 2460, 'AA': 10, 'TT': 10, 'AT': 10, 'TA': 10},
      {'A': 2460, 'T': 2460, 'AA': 10, 'AT': 10, 'TA': 10, 'TT': 10},
      {'A': 2460, 'T': 2460, 'TT': 10, 'TA': 10, 'AT': 10, 'AA': 10},
      {'A': 2460, 'T': 2460, 'TA': 10, 'AT': 10, 'AA': 10, 'TT': 10},
      {'A': 2460, 'T': 2460, 'TT': 10, 'AA': 10, 'AT': 10, 'TA': 10},
      {'A': 2460, 'T': 2460, 'AT': 10, 'TT': 10, 'AA': 10, 'TA': 10}]
    filepath = kdi.utils._create_typical_strand_trajectory_section_complexes_filepath('9999_99_99__99_99_99')
    steps, times, sequence_number_trajectory = kdi.read.steps_and_times_and_sequence_trajectory_from_complexes_txt(filepath)
    assert_equal(steps,desired_steps)
    assert_equal(times,desired_times)
    assert_equal(sequence_number_trajectory,desired_sequence_number_trajectory)

def _assert_desired_motif_vector(actual : np.ndarray,
        motifs : list,
        value : int,
        alphabet : list,
        motiflength : int):
    desired = np.zeros(actual.shape)
    for motif in motifs:
        desired[kdi.read._transform_motif_string_to_index_tuple(motif,alphabet)] += value
    assert_equal(actual,desired)

def test__transform_sequence_vector_into_motif_vector():
    sequence_vector = {'ATTATA':100,'A':10}
    alphabet = ['A','T']
    motiflength = 5

    one_mer = 'A'
    beginning = 'ATTA'
    end = 'TATA'
    continuations = ['ATTAT','TTATA']

    motif_vector = kdi.read._transform_sequence_vector_into_motif_vector(sequence_vector,alphabet, motiflength)
    mvd = motif_vector.motifs.val

    motif_categories = kdi.domains._return_motif_categories(motiflength)
    _assert_desired_motif_vector(mvd[motif_categories[0]],
            [one_mer,],
        10,alphabet,motiflength)
    _assert_desired_motif_vector(mvd[motif_categories[-3]],
        [beginning,],
        100,alphabet,motiflength)
    _assert_desired_motif_vector(mvd[motif_categories[-1]], [end,],
        100,alphabet, motiflength)
    _assert_desired_motif_vector(mvd[motif_categories[-2]], continuations,
        100,alphabet, motiflength)
