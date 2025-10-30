import morsaik as kdi
import jax.numpy as jnp

def test_infer_total_mass():
    strand_trajectory_id='9999_99_99__99_99_99'
    motiflengths = [2,3,4,5]

    for motiflength in motiflengths:
        smt = kdi.get.strand_motifs_trajectory(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                simulations_run_no=0
                )
        desired_total_mass = 2*2460 + 4*2*10
        actual_total_mass = kdi.infer.total_mass(
                kdi.extract_initial_motif_vector_from_motif_trajectory(smt)
                )
        assert actual_total_mass == desired_total_mass

def test_infer_total_mass_trajectory():
    strand_trajectory_id='9999_99_99__99_99_99'
    motiflengths = [2,3,4,5]

    for motiflength in motiflengths:
        smt = kdi.get.strand_motifs_trajectory(
                motiflength,
                strand_trajectory_id,
                param_file_no=0,
                simulations_run_no=0
                )
        desired_total_mass_trajectory = (2*2460 + 4*2*10)*jnp.ones(smt.times.size)
        actual_total_mass_trajectory = kdi.infer.total_mass_trajectory(smt)
        assert jnp.all(actual_total_mass_trajectory == desired_total_mass_trajectory)

def test_infer_total_mass_of_motif_concentration_trajectory_array():
    number_of_letters = motiflength = 4
    number_of_motifs = int(jnp.sum(number_of_letters**jnp.arange(1,motiflength+1))+number_of_letters**(motiflength-1))
    motif_vec = jnp.arange(1,number_of_motifs+1)
    masses = jnp.ones(motif_vec.shape).at[slice(4,20)].add(1)
    masses = masses.at[slice(84+4**4,None)].add(1)
    masses = masses.flatten()
    assert jnp.allclose(
            jnp.vecdot(masses,motif_vec).flatten(),
            (kdi.infer.total_mass_of_motif_concentration_trajectory_array(motif_vec.reshape(1,-1),number_of_letters,motiflength)).flatten()
            )
