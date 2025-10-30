import morsaik as kdi
import jax.numpy as jnp
from jax import random

def setup_c(test_motifs=[(0,1,0,0),(0,2,3,0),(0,1,3,2),(1,3,2,4),(3,2,4,0)],
            test_concentration=0.1,
            nol=4,
            motiflength = 4):
    c = jnp.zeros([number_of_letters+1,]*motiflength)
    for motif in test_motifs:
        c = c.at[motif].set(test_concentration)
    return c

def total_mass(c):
    return jnp.sum(c[0,1:,:,0]) + jnp.sum(c[:,1:,1:,1:])+ 2*jnp.sum(c[1:,1:,1:,0]) + jnp.sum(c[0,1:,1:,0])

def test_break_continuations():
    number_of_letters = 4
    motiflength = 4
    monomer_motif = (0,1,0,0)
    dimer_motif = (0,2,3,0) 
    beginning_motif = (0,1,3,2)
    continuing_motif =(1,3,2,4)
    end_motif =(3,2,4,0)
    breakage_rate_constant = 1.e-5
    pseudo_count_concentration = 1e-12
    test_concentration = 0.1
    test_motifs = [monomer_motif,dimer_motif,beginning_motif,continuing_motif,end_motif]

    c = jnp.zeros([number_of_letters+1,]*motiflength)
    for motif in test_motifs:
        c = c.at[motif].set(test_concentration)
    fourmer_logc = jnp.log(c.at[c<pseudo_count_concentration].set(pseudo_count_concentration))
    breakage_rates = kdi.infer.fourmer_breakage_rates(
            fourmer_logc,
            breakage_rate_constant,
            pseudo_count_concentration=pseudo_count_concentration
            )
    # we have the following reactions:
    # (0,2,3,0) -> (0,2,0,0) + (0,3,0,0)
    # (0,1,3,2,4,0) -> (0,1,0,0) + (0,3,2,4,0)
    # (0,1,3,2,4,0) -> (0,1,3,0) + (0,2,4,0)
    # (0,1,3,2,4,0) -> (0,1,3,2,0) + (0,4,0,0)
    desired = jnp.zeros(breakage_rates.shape)
    desired = desired.at[0,2,3,0].set(-breakage_rate_constant)
    desired = desired.at[0,2,0,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[0,2,3,0]-fourmer_logc[0,2,0,0]))
    desired = desired.at[0,3,0,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[0,2,3,0]-fourmer_logc[0,3,0,0]))
    desired = desired.at[0,1,0,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[0,1,3,2]-fourmer_logc[0,1,0,0]))
    desired = desired.at[0,3,2,4].set(breakage_rate_constant*jnp.exp(fourmer_logc[1,3,2,4]-fourmer_logc[0,3,2,4]))
    desired = desired.at[0,1,3,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[0,1,3,2]-fourmer_logc[0,1,3,0]))
    desired = desired.at[0,2,4,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[3,2,4,0]-fourmer_logc[0,2,4,0]))
    desired = desired.at[1,3,2,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[1,3,2,4]-fourmer_logc[1,3,2,0]))
    desired = desired.at[0,4,0,0].set(breakage_rate_constant*jnp.exp(fourmer_logc[3,2,4,0]-fourmer_logc[0,4,0,0]))
    desired = desired.at[1,3,2,4].set(-3*breakage_rate_constant)
    desired = desired.at[3,2,4,0].set(-2*breakage_rate_constant)
    desired = desired.at[0,1,3,2].set(-2*breakage_rate_constant)
    ebr = breakage_rates*jnp.exp(fourmer_logc)
    edr = desired*jnp.exp(fourmer_logc)
    assert jnp.allclose(
            ebr,
            edr,
            rtol=1.e-4, 
            atol=pseudo_count_concentration
            )
    assert jnp.allclose(
            total_mass(ebr),
            0.,
            atol=pseudo_count_concentration)

def test_break_dimers():
    pseudo_count_concentration = 1.e-12
    brc = .1
    key = random.PRNGKey(42)
    key, key1 = random.split(key)
    fourmer_logc = jnp.log(random.uniform(key1,shape=(5,5,5,5)))
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    fourmer_c = jnp.exp(fourmer_logc)
    ebrc = jnp.zeros(fourmer_logc.shape)
    ebrc = ebrc.at[0,1,2,0].set(brc)
    br = kdi.infer.fourmer_breakage_rates(fourmer_logc,ebrc,pseudo_count_concentration=pseudo_count_concentration)
    assert jnp.allclose(br[0,1,0,0]*fourmer_c[0,1,0,0]/fourmer_c[0,1,2,0],brc,atol=pseudo_count_concentration)
    assert jnp.allclose(br[0,2,0,0]*fourmer_c[0,2,0,0]/fourmer_c[0,1,2,0],brc,atol=pseudo_count_concentration)
    assert jnp.allclose(br[0,1,2,0],-brc,atol=pseudo_count_concentration)
    brr = br.at[0,1,0,0].set(0.)
    brr = brr.at[0,2,0,0].set(0.)
    brr = brr.at[0,1,2,0].set(0.)
    assert jnp.allclose(brr,0.,atol=pseudo_count_concentration)

def setup_consistent_random_fourmer_population(number_of_letters : int = 4):
    nol = number_of_letters
    key = random.PRNGKey(42)
    key, key1 = random.split(key)
    fourmer_c = jnp.zeros((nol+1,)*4)
    # draw strands
    fourmer_c = fourmer_c.at[0,1:,:,0].set(random.uniform(key1,shape=(nol,nol+1)))
    key, key2 = random.split(key)
    # draw beginnings
    fourmer_c = fourmer_c.at[0,1:,1:,1:].set(random.uniform(key2,shape=(nol,)*3))
    key, key3 = random.split(key)
    # draw continuations
    continuation_distribution = random.uniform(key3,shape=(nol,)*4)
    continuation_distribution = continuation_distribution.at[:,:,:,:].divide(
            jnp.sum(continuation_distribution, axis=-1)[:,:,:,None]
            )
    fourmer_c = fourmer_c.at[1:,1:,1:,1:].set(
            continuation_distribution.at[:].set(continuation_distribution*fourmer_c[0,1:,1:,1:,None])
    )
    # draw ends
    fourmer_c = fourmer_c.at[1:,1:,1:,0].set(
            jnp.sum(fourmer_c[1:,1:,1:,1:],axis=0)
            )
    return fourmer_c

def test_break_left():
    pseudo_count_concentration = 1.e-12
    brc = .1
    fourmer_c = setup_consistent_random_fourmer_population(number_of_letters=4)
    fourmer_logc = jnp.log(fourmer_c)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    ebrc = jnp.zeros(fourmer_logc.shape)
    ebrc = ebrc.at[0,1,1,2].set(brc)
    br = kdi.infer.fourmer_breakage_rates(fourmer_logc,ebrc,pseudo_count_concentration=pseudo_count_concentration)
    rcf = fourmer_c[1,1,2,0]/jnp.sum(fourmer_c[1,1,2,:],axis=-1)
    desired = brc*fourmer_c[0,1,1,2]*rcf/fourmer_c[0,1,2,0]
    assert jnp.allclose(br[0,1,2,0],desired,atol=pseudo_count_concentration)
    desired = -brc*fourmer_c[0,1,1,2]*rcf/fourmer_c[1,1,2,0]
    assert jnp.allclose(br[1,1,2,0],desired,atol=pseudo_count_concentration)
    ebr = br*fourmer_c
    assert jnp.allclose(total_mass(ebr),0.,atol=pseudo_count_concentration)

def test_break_central():
    pseudo_count_concentration = 1.e-12
    brc = .1
    fourmer_c = setup_consistent_random_fourmer_population(number_of_letters=4)
    fourmer_logc = jnp.log(fourmer_c)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    ebrc = jnp.zeros(fourmer_logc.shape)
    ebrc = ebrc.at[1,1,2,2].set(brc)
    br = kdi.infer.fourmer_breakage_rates(fourmer_logc,ebrc,pseudo_count_concentration=pseudo_count_concentration)
    desired = -brc
    assert jnp.allclose(br[1,1,2,2],desired,atol=pseudo_count_concentration)
    ebr = br*fourmer_c
    assert jnp.allclose(total_mass(ebr),0.,atol=pseudo_count_concentration)

def test_break_right():
    pseudo_count_concentration = 1.e-12
    brc = .1
    fourmer_c = setup_consistent_random_fourmer_population(number_of_letters=4)
    fourmer_logc = jnp.log(fourmer_c)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    ebrc = jnp.zeros(fourmer_logc.shape)
    ebrc = ebrc.at[1,1,2,0].set(brc)
    br = kdi.infer.fourmer_breakage_rates(fourmer_logc,ebrc,pseudo_count_concentration=pseudo_count_concentration)
    lcf = fourmer_c[0,1,1,2]/jnp.sum(fourmer_c[:,1,1,2],axis=0)
    desired = brc*fourmer_c[1,1,2,0]*lcf/fourmer_c[0,1,1,0]
    assert jnp.allclose(br[0,1,1,0],desired,atol=pseudo_count_concentration)
    desired = -brc*fourmer_c[1,1,2,0]*lcf/fourmer_c[0,1,1,2]
    assert jnp.allclose(br[0,1,1,2],desired,atol=pseudo_count_concentration)
    ebr = br*fourmer_c
    assert jnp.allclose(total_mass(ebr),0.,atol=pseudo_count_concentration)

def test_mass_conservation():
    pseudo_count_concentration = 1.e-12
    key = random.PRNGKey(42)
    key, key1 = random.split(key)
    fourmer_c = setup_consistent_random_fourmer_population(number_of_letters=4)
    fourmer_logc = jnp.log(fourmer_c)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    effective_breakage_rate_constants = .1
    breakage_rates = kdi.infer.fourmer_breakage_rates(
            fourmer_logc,
            effective_breakage_rate_constants,
            pseudo_count_concentration
            )
    ebr = breakage_rates*jnp.exp(fourmer_logc)
    assert jnp.allclose(total_mass(ebr),0.,atol=pseudo_count_concentration)
