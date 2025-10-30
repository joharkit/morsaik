import morsaik as kdi
import jax.numpy as jnp
import pytest

@pytest.mark.skip
def test_motif_production_rates_from_motif_concentrations_array_and_motif_production_rate_constants():
    motiflength = 4
    number_of_letters = 4
    motifs_concentrations_array = (1+jnp.arange(404))/404
    motif_production_rate_constants = jnp.ones((5,4,4,5,5,4,4,5))
    motif_production_rate_constants = jnp.arange(1,motif_production_rate_constants.size+1, dtype=float).reshape(motif_production_rate_constants.shape)

    collisions_vector = kdi.infer.ligation_spot_formations_from_motifs_array(motifs_concentrations_array)
    extended_motif_production_rate_constants = kdi.infer._extend_motif_production_rate_constants_array_to_collisions_format(motif_production_rate_constants,maximum_ligation_window_length=motiflength,motiflength=motiflength,number_of_letters=number_of_letters)
    motif_production_rates = extended_motif_production_rate_constants*collisions_vector
    motif_production_transition_kernel_matrix = kdi.infer.motif_production_transition_kernel_matrix()
    motif_extensions = motif_production_transition_kernel_matrix.reshape((-1,motif_production_rates.size)) @ motif_production_rates.flatten()
    mass_difference = kdi.infer.total_mass_of_motif_concentration_trajectory_array(motif_extensions.reshape(1,-1))
    relative_error = mass_difference/jnp.min(jnp.abs(motif_extensions))
    assert jnp.allclose(relative_error,0.)

def test_motif_production_rates_array_from_motif_production_rate_constants_array_and_motif_concentrations_array():
    timesteps = 4
    t_max = 3.
    initial_a_concentration = .1
    initial_aa_concentration = .1
    mprc_value = .1

    motif_logconcentrations_trajectory = jnp.zeros((timesteps,)+(3,2,3,3))
    motif_logconcentrations_trajectory = motif_logconcentrations_trajectory.at[:,0,0,0,0].set(initial_a_concentration)
    motif_logconcentrations_trajectory = motif_logconcentrations_trajectory.at[:,0,0,1,0].set(initial_aa_concentration)
    # unzero
    motif_logconcentrations_trajectory = motif_logconcentrations_trajectory.at[motif_logconcentrations_trajectory==0.].add(1.e-12)
    motif_logconcentrations_trajectory = jnp.log(motif_logconcentrations_trajectory)

    motif_concentration_trajectory_times_array = jnp.linspace(0,t_max,timesteps)
    # setup motif concentration trjaectory times array

    motif_production_rate_constants = jnp.zeros((3,2,2,3,3,2,2,3))
    motif_production_rate_constants = motif_production_rate_constants.at[0,:,:,0,0,:,:,0].add(mprc_value)
    exposure = kdi.infer.collisions_from_motif_concentration_trajectory_array_and_collision_rate_constants_array(
            motif_logconcentrations_trajectory,
            motif_concentration_trajectory_times_array=motif_concentration_trajectory_times_array,
            concentrations_are_logarithmised = True
            )
    desired = motif_production_rate_constants.reshape(-1)*exposure
    assert jnp.allclose(
            desired.reshape((3,2,2,3,3,2,2,3)),
            kdi.infer.motif_production_rates_array_from_motif_production_rate_constants_array_and_motif_concentrations_array(
            motif_production_rate_constants,
            motif_logconcentrations_trajectory,
            motif_concentration_trajectory_times_array
            )
    )
