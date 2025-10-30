import morsaik as kdi
import numpy as np
import pytest
from numpy.testing import assert_equal

def test_effective_ligation_rates_from_parameters():
    strand_trajectory_id = '9999_99_99__99_99_99'
    complements = [1,0]
    strand_trajectory_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    hp, lp = kdi.infer.separate_hybridization_and_ligation_parameters(strand_trajectory_parameters)

    dissociation_constant = kdi.infer.dissociation_constant_from_strand_reactor_parameters(strand_trajectory_parameters['l_critical'],lp,complements,hp)
    effective_ligation_rate_constants = kdi.infer.effective_ligation_rates_from_parameters(strand_trajectory_parameters, complements)

    assert effective_ligation_rate_constants[(0,)*8] == np.exp(strand_trajectory_parameters['l_critical']*hp['dG_4_2Match_mean'])*lp['stalling_factor_first']**2/dissociation_constant[(0,)*8]
    assert effective_ligation_rate_constants[(0,)*4+(0,1,1,0)] == np.exp(strand_trajectory_parameters['l_critical']*hp['dG_4_2Match_mean'])/dissociation_constant[(0,)*4+(0,1,1,0)]
    assert kdi.infer.effective_ligation_rates_from_parameters(strand_trajectory_parameters, [1,0])[(0,0,1,0)+(0,1,1,0)] == np.exp(strand_trajectory_parameters['l_critical']*hp['dG_4_2Match_mean'])*lp['stalling_factor_first']/dissociation_constant[(0,0,1,0)+(0,1,1,0)]
    assert kdi.infer.effective_ligation_rates_from_parameters(strand_trajectory_parameters, [1,0])[(1,0,1,0)+(0,1,1,1)] == np.exp(strand_trajectory_parameters['l_critical']*hp['dG_4_2Match_mean'])*lp['stalling_factor_first']/dissociation_constant[(1,0,1,0)+(0,1,1,1)]
