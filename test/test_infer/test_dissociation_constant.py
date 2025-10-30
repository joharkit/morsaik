import morsaik as kdi
import numpy as np

def test_dissociation_constant_from_strand_reactor_parameters():
    strand_trajectory_id = '9999_99_99__99_99_99'
    strand_trajectory_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    hp, lp = kdi.infer.separate_hybridization_and_ligation_parameters(strand_trajectory_parameters)
    dissociation_constant = kdi.infer.dissociation_constant_from_strand_reactor_parameters(10,lp,[1,0],hp)
    assert dissociation_constant[0,1,1,0,0,1,1,0]==np.exp(hp['dG_4_0Match'])
    assert dissociation_constant[0,1,0,0,0,1,1,0]==np.exp(hp['dG_4_1Match'])
    assert dissociation_constant[0,0,0,0,0,1,1,0]==np.exp(hp['dG_4_2Match_mean']-hp['ddG_4_2Match_alternating'])
    assert dissociation_constant[0,0,1,0,0,0,1,0]==np.exp(hp['dG_4_2Match_mean']+hp['ddG_4_2Match_alternating'])

def test_ligation_rate_constant_from_strand_reactor_parameters():
    strand_trajectory_id = '9999_99_99__99_99_99'
    strand_trajectory_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    hp, lp = kdi.infer.separate_hybridization_and_ligation_parameters(strand_trajectory_parameters)
    ligation_rate_constant = kdi.infer.ligation_rate_constant_from_strand_reactor_parameters(hp,lp,[0,2,1],strand_trajectory_parameters['l_critical'])
    assert ligation_rate_constant[0,1,1,0,0,0,0,0]==np.exp(strand_trajectory_parameters['l_critical']*strand_trajectory_parameters['dG_4_2Match_mean'])
    assert ligation_rate_constant[0,1,1,0,0,1,1,0]==np.exp(strand_trajectory_parameters['l_critical']*strand_trajectory_parameters['dG_4_2Match_mean'])*lp['stalling_factor_first']**2
    assert ligation_rate_constant[1,1,1,1,1,1,1,1]==np.exp(strand_trajectory_parameters['l_critical']*strand_trajectory_parameters['dG_4_2Match_mean'])*(lp['stalling_factor_first']*lp['stalling_factor_second'])**2

def test_energy_continuous_block():
    strand_trajectory_id = "9999_99_99__99_99_99"
    strand_trajectory_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    assert kdi.infer.energy_continuous_block(2,1,1,2,[0,2,1],strand_trajectory_parameters) == strand_trajectory_parameters['dG_4_2Match_mean']
    assert kdi.infer.energy_continuous_block(2,1,1,1,[0,2,1],strand_trajectory_parameters) == strand_trajectory_parameters['dG_4_1Match']
    assert kdi.infer.energy_continuous_block(1,1,1,1,[0,2,1],strand_trajectory_parameters) == strand_trajectory_parameters['dG_4_0Match']

def test_energy_continuous_block():
    strand_trajectory_id = "9999_99_99__99_99_99"
    strand_trajectory_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id)
    hp, _ = kdi.infer.separate_hybridization_and_ligation_parameters(strand_trajectory_parameters)
    ecp = kdi.infer._energy_continuous_blocks((3,2,2,3),[1,0],hp)
    assert ecp[(0,)*8]==strand_trajectory_parameters['dG_4_0Match']
    assert ecp[(1,)*8]==3*strand_trajectory_parameters['dG_4_0Match']
    assert ecp[(0,1,1,0)+(0,)*4]==strand_trajectory_parameters['dG_4_2Match_mean']
    assert ecp[(0,1,1,0)+(0,1,0,0)]==strand_trajectory_parameters['dG_4_1Match']
    assert ecp[(0,1,1,0)+(0,1,1,0)]==strand_trajectory_parameters['dG_4_0Match']
    assert ecp[(1,1,1,0)+(0,1,1,0)]==strand_trajectory_parameters['dG_4_0Match']+strand_trajectory_parameters['dG_3_0Match']
    assert ecp[(1,1,1,0)+(0,1,1,1)]==2*strand_trajectory_parameters['dG_4_0Match']
    assert ecp[(1,1,1,0)+(1,1,1,0)]==strand_trajectory_parameters['dG_4_0Match']+2*strand_trajectory_parameters['dG_3_0Match']
    assert ecp[(1,1,1,0)+(1,1,1,0)]==strand_trajectory_parameters['dG_4_0Match']+2*strand_trajectory_parameters['dG_3_0Match']
    assert ecp[(1,1,1,0)+(1,1,0,0)]==strand_trajectory_parameters['dG_4_1Match']+strand_trajectory_parameters['dG_3_1Match_mean']+strand_trajectory_parameters['dG_3_0Match']
