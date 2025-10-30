import morsaik as kdi
import jax.numpy as jnp

def test_infer_mass_correction_rates():
    number_of_letters = 2
    motiflength = 4
    soft_reactant_threshold_concentration = 1.e-12

    initial_concentration_vector = soft_reactant_threshold_concentration*jnp.ones((number_of_letters+1,)*motiflength)
    updated_concentration_vector = soft_reactant_threshold_concentration*jnp.ones((number_of_letters+1,)*motiflength)

    initial_concentration_vector = initial_concentration_vector.at[0,1,1,0].set(.1)
    updated_concentration_vector = updated_concentration_vector.at[0,1,1,0].set(.2)

    initial_logc_vector = jnp.log(initial_concentration_vector)
    updated_logc_vector = jnp.log(updated_concentration_vector)

    actual_mcr = kdi.infer.mass_correction_rates(
            initial_logc_vector,
            updated_logc_vector
            )
    """
    Delta_c_i = 0.2
    Delta_c_i * A_ij c_j
    0.2*2*0.2 for i==1, j==[0,1,1,0]
    """
    desired_mcr = jnp.zeros(initial_concentration_vector.shape)
    desired_mcr = desired_mcr.at[0,1,1,0].set(-0.2*2*0.2)

    assert jnp.allclose(actual_mcr,desired_mcr)

def test_infer_mass_correction_rates_nonending_strand():
    """
    Delta_c_i = .12+2*0.09 - 3*0.1 = 0
    -> only correction of nonending strands

    Delta_beginning_ending = .12 - .09 = .03 = 3.e-2
    => nsr[0,1,1,1] = -3.e-2*1.2e-1 = -3.6e-3
    nsr[1,1,1,0] = 3.e-2*9.e-2 = 2.7e-3
    """
    number_of_letters = 2
    motiflength = 4
    soft_reactant_threshold_concentration = 1.e-12

    initial_concentration_vector = soft_reactant_threshold_concentration*jnp.ones((number_of_letters+1,)*motiflength)
    updated_concentration_vector = soft_reactant_threshold_concentration*jnp.ones((number_of_letters+1,)*motiflength)

    initial_concentration_vector = initial_concentration_vector.at[0,1,1,1].set(.1)
    initial_concentration_vector = initial_concentration_vector.at[1,1,1,0].set(.1)
    updated_concentration_vector = updated_concentration_vector.at[0,1,1,1].set(.12)
    updated_concentration_vector = updated_concentration_vector.at[1,1,1,0].set(.09)

    initial_logc_vector = jnp.log(initial_concentration_vector)
    updated_logc_vector = jnp.log(updated_concentration_vector)

    actual_mcr = kdi.infer.mass_correction_rates(
            initial_logc_vector,
            updated_logc_vector
            )
    desired_mcr = jnp.zeros(initial_concentration_vector.shape)
    desired_mcr = desired_mcr.at[0,1,1,1].set(-3.6e-3)
    desired_mcr = desired_mcr.at[1,1,1,0].set(2.7e-3)

    print(f"{actual_mcr = }")
    print(f"{desired_mcr = }")
    assert jnp.allclose(actual_mcr,desired_mcr)
