import morsaik as kdi
import jax.numpy as jnp
import pytest
from itertools import product as iterprod

@pytest.mark.skip
def test_extend_motif_production_rate_constants_array_to_collisions_format():
    motiflength = 4
    number_of_letters = 4
    maximum_ligation_window_length = 4
    motif_production_rate_constants = jnp.ones(
            (number_of_letters+1,)*(motiflength-motiflength//2-1)+(number_of_letters,)*2+(number_of_letters+1,)*(motiflength//2-1)
            + (number_of_letters+1,)*(motiflength//2-1)+(number_of_letters,)*2+(number_of_letters+1,)*(motiflength-motiflength//2-1)
            )
    motif_production_rate_constants = jnp.arange(1,motif_production_rate_constants.size+1).reshape(motif_production_rate_constants.shape)
    ligation_spot_formations =  kdi.infer._extend_motif_production_rate_constants_array_to_collisions_format(
            motif_production_rate_constants,
            maximum_ligation_window_length,
            motiflength,
            number_of_letters
            )
    reactant_indices = jnp.concatenate((jnp.zeros(1),jnp.cumsum(number_of_letters**jnp.arange(1,motiflength))), dtype=int)
    reactant_slices = [slice(reactant_indices[ii],reactant_indices[ii+1]) for ii in range(len(reactant_indices)-1)]
    left_indices = [(0,)*int(motiflength-motiflength//2-strandlength)+(slice(1,number_of_letters+1),)*int(strandlength-1) + (slice(None),) for strandlength in jnp.arange(1,motiflength)]
    right_indices = [(slice(None),) + (slice(1,number_of_letters+1),)*int(strandlength-1) + (0,)*int(motiflength//2-strandlength) for strandlength in jnp.arange(1,motiflength)]
    if motiflength == 4:
        cat_indices = [((0,) + (slice(None),)*2 + (slice(1,None),)*int(strandlength-2) + (0,)*int(motiflength-1-strandlength)) for strandlength in range(2,motiflength)] + [(slice(1,None),slice(None),slice(None),slice(1,None))] + [(slice(1,None),)+(slice(None),)*2 + (0,)]
    else:
        raise NotImplementedError(f"only motiflength = 4, but {motiflength = }")
    template_indices = jnp.concatenate((jnp.zeros(1, dtype=int),jnp.cumsum(number_of_letters**jnp.arange(2,motiflength+1)),jnp.ones(1)*(jnp.sum(number_of_letters**jnp.arange(2,motiflength+1))+number_of_letters**(motiflength-1))),dtype=int)
    template_slices = [slice(template_indices[ii],template_indices[ii+1]) for ii in range(len(template_indices)-1)]
    for ii,jj in iterprod(range(len(reactant_slices)-1), repeat=2):
        for kk in range(len(template_slices)):
            assert bool(jnp.all(ligation_spot_formations[(reactant_slices[ii],reactant_slices[jj],template_slices[kk])].flatten() == motif_production_rate_constants[left_indices[ii]+right_indices[jj]+cat_indices[kk]].flatten()))

    assert ligation_spot_formations[0,4,335]==motif_production_rate_constants[0,0,0,1,4,3,3,4]

@pytest.mark.skip
def test_motif_production_transition_kernel_from_motif_production_rate_constants_array():
    number_of_letters = motiflength = maximum_ligation_window_length = 4
    number_of_reactants = int(jnp.sum(number_of_letters**jnp.arange(1,motiflength)))
    number_of_templates = int(jnp.sum(number_of_letters**jnp.arange(1,motiflength+1))+number_of_letters**(motiflength-1)-number_of_letters)
    motif_collisions = jnp.zeros((number_of_reactants,number_of_reactants,number_of_templates))
    motif_collisions = jnp.arange(1,1+motif_collisions.size).reshape(motif_collisions.shape)
    # motif_diff
    motif_production_transition_kernel_matrix = kdi.infer.motif_production_transition_kernel_matrix(number_of_letters, motiflength, maximum_ligation_window_length)
    motif_diff = motif_production_transition_kernel_matrix.reshape((-1,motif_collisions.size)) @ motif_collisions.flatten()
    total_mass_diff = kdi.infer.total_mass(motif_diff)
    assert jnp.allclose(total_mass_diff,0)

    motif_production_rate_constants = jnp.ones((number_of_letters+1,number_of_letters,number_of_letters,number_of_letters+1,)*2)
    motif_production_rate_constants = jnp.arange(1,motif_production_rate_constants.size+1, dtype=float).reshape(motif_production_rate_constants.shape)
    motif_production_transition_kernel = kdi.infer.motif_production_transition_kernel_from_motif_production_rate_constants_array(
            motif_production_rate_constants = motif_production_rate_constants
            )

    motif_diff = kdi.infer.motif_production_transition_kernel_from_motif_production_rate_constants_array
    motif_production_transition_kernel
    # test mass conservation
    total_mass_trajectory = kdi.infer.total_mass_of_motif_concentration_trajectory_array(motif_trajectory)
    assert jnp.allclose(total_mass_trajectory, total_mass_trajectory[0])
    # monomer rates
    assert bool(motif_diff[0] == -jnp.sum(motif_collisions[0,:,:])-jnp.sum(motif_collisions[:,0,:]))
    # dimer rates
    assert bool(motif_diff[4] == -jnp.sum(motif_collisions[4,:,:])-jnp.sum(motif_collisions[:,4,:])+jnp.sum(motif_collisions[0,0,:]))
    # beginning rates
    assert bool(motif_diff[20] == jnp.sum(motif_collisions[0,4])+jnp.sum(motif_collisions[0,20:].reshape((4,4,-1))[0,0])+jnp.sum(motif_collisions[4,0])+jnp.sum(motif_collisions[4,4:20].reshape((4,-1))[0])+jnp.sum(motif_collisions[4,20:].reshape((4,-1))[0])-jnp.sum(motif_collisions[:,4+16,:]))
    # continuation rates 
    assert bool(motif_diff[84] == (
        jnp.sum(motif_collisions[0,20])+jnp.sum(motif_collisions[4,4])+
        jnp.sum(motif_collisions[4,20:].reshape((16,-1))[0])+
        jnp.sum(motif_collisions[4:20,20:].reshape((4,256,-1))[:,0])+
        jnp.sum(motif_collisions[20,0])+
        jnp.sum(motif_collisions[20:,4:20].reshape((256,-1))[0])+
        jnp.sum(motif_collisions[20:,4].reshape((4,16,-1))[:,0])+
        jnp.sum(motif_collisions[20:,20:].reshape((256,-1))[0])+
        jnp.sum(motif_collisions[20:,20:].reshape((4,256,-1))[:,0])+
        jnp.sum(motif_collisions[20:,20:].reshape((16,256,-1))[:,0]))
                )
    # ending rates
    assert bool(
            motif_diff[-64] == (
                jnp.sum(motif_collisions[4,0].flatten())+
                jnp.sum(motif_collisions[20:,0].reshape((4,16,-1))[:,0].flatten())+
                jnp.sum(motif_collisions[0,4].flatten())+
                jnp.sum(motif_collisions[4:20,4].reshape((4,4,-1))[:,0].flatten())+
                jnp.sum(motif_collisions[20:,4].reshape((16,4,-1))[:,0].flatten())-
                jnp.sum(motif_collisions[20].flatten())
                )
            )
    # test exemplaric reactions
