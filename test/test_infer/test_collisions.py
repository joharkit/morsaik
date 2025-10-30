import morsaik as kdi
import jax.numpy as jnp

def test_collisions_from_motif_concentration_trajectory_array_and_collision_rate_constants_array():
    timesteps = 4
    t_max = 3.
    initial_a_concentration = .1
    initial_aa_concentration = .1
    # setup motif concentration trajectory array
    motif_concentration_trajectory_array = jnp.zeros((timesteps,)+(3,2,3,3))
    motif_concentration_trajectory_array = motif_concentration_trajectory_array.at[:,0,0,0,0].set(initial_a_concentration)
    motif_concentration_trajectory_array = motif_concentration_trajectory_array.at[:,0,0,1,0].set(initial_aa_concentration)
    # setup motif concentration trjaectory times array
    motif_concentration_trajectory_times_array = jnp.linspace(0,t_max,timesteps)
    # call collisions_from_motif_concentration_trajectory_array_and_collision_rate_constants_array
    actual = kdi.infer.collisions_from_motif_concentration_trajectory_array_and_collision_rate_constants_array(
            motif_concentration_trajectory_array,
            motif_concentration_trajectory_times_array,
            concentrations_are_logarithmised=False
            )
    # manually calculate collisions
    collisions = jnp.zeros((3,3,2,2,3,3)+(3,2,2,3))
    # 0,0,a-a,0,0-0,a,a,0
    collisions = collisions.at[(0,)*10].add(t_max*initial_a_concentration*initial_a_concentration*initial_aa_concentration)
    # 0,0,a-a,a,0-0,a,a,0
    collisions = collisions.at[(0,0,0)+(0,1,0)+(0,0,0,0)].add(t_max*initial_a_concentration*initial_aa_concentration*initial_aa_concentration)
    # 0,a,a-a,0,0-0,a,a,0
    collisions = collisions.at[(0,1,0)+(0,0,0)+(0,0,0,0)].add(t_max*initial_aa_concentration*initial_a_concentration*initial_aa_concentration)
    # 0,a,a-a,a,0-0,a,a,0
    collisions = collisions.at[(0,1,0)+(0,1,0)+(0,0,0,0)].add(t_max*initial_aa_concentration*initial_aa_concentration*initial_aa_concentration)
    collisions = jnp.sum(collisions, axis=(0,5))
    desired = collisions.reshape(-1)
    # check output
    assert jnp.allclose(actual, desired)

def test_infer_motifs_collisions_array_from_motifs_array():
    motiflengths = [4]#[2,3,4,5,6]
    nol = 4
    for motiflength in motiflengths:
        number_of_motifs = nol**(motiflength-1)
        for strandlength in range(1,motiflength):
            number_of_motifs += nol**strandlength
        motifs_array = jnp.arange(number_of_motifs)
        actual_collisions_array = kdi.infer.motifs_collisions_array_from_motifs_array(motifs_array)
        desired_collisions_array = (motifs_array.reshape(-1,1)*motifs_array.reshape(1,-1)).reshape(-1,1)*motifs_array.reshape(1,-1)
        assert bool(jnp.all(jnp.equal(actual_collisions_array.flatten(), desired_collisions_array.flatten())))

def test_infer_ligation_spot_formations_from_motifs_array():
    motiflengths = [4]#[2,3,4,5,6]
    nol = 4
    collision_order = 3
    for motiflength in motiflengths:
        number_of_motifs = jnp.sum(nol**jnp.arange(1,motiflength+1)) + nol**(motiflength-1)
        motifs_array = jnp.arange(number_of_motifs)
        desired_collisions_array = kdi.infer.motifs_collisions_array_from_motifs_array(motifs_array)
        desired_collisions_array = desired_collisions_array.reshape((number_of_motifs,)*collision_order)
        # not all collisions enable adjacent hybridization
        number_of_continuations = nol**motiflength
        number_of_beginnings = number_of_endings = nol**(motiflength-1)
        # fore motif must end
        desired_collisions_array = jnp.concatenate([
            desired_collisions_array[:-(number_of_beginnings + number_of_endings + number_of_continuations),:,:],
            desired_collisions_array[-number_of_endings:,:,:]
            ],
                                                   axis = 0)
        # rear motif must begin
        desired_collisions_array = desired_collisions_array[:,:-(number_of_continuations+number_of_endings),:]
        # template must have at least one covalent bond
        desired_collisions_array = desired_collisions_array[:,:,nol:]
        actual_collisions_array = kdi.infer.ligation_spot_formations_from_motifs_array(
                motifs_array,
                number_of_letters = nol,
                motiflength = motiflength)
        assert bool(jnp.all(jnp.equal(actual_collisions_array.flatten(), desired_collisions_array.flatten())))
        assert actual_collisions_array.flatten()[0] == motifs_array[0]**2*motifs_array[1]
        assert actual_collisions_array.flatten()[-1] == motifs_array[-(number_of_continuations+number_of_endings)-1]*motifs_array[-1]*motifs_array[-1]
