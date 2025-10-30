import morsaik as kdi
import jax.numpy as jnp
from jax import random
from itertools import product as iterprod
import pytest
from copy import deepcopy

def test_compute_extended_end_motif_reaction_rates():
    motiflength = 4
    number_of_letters = 4
    pseudo_count_concentration = 1.e-12
    logc_val = -3.
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    e4pr = kdi.infer._initialize_empty_fourmer_production_rates(number_of_letters)
    log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))
    log_reaction_rate_constants = log_reaction_rate_constants.at[0,0,0,0,0,1,1,0].set(0.)
    lrr = kdi.infer._compute_extended_end_motif_reaction_logc_rates(fourmer_logc,log_reaction_rate_constants)
    assert jnp.allclose(lrr.at[0,0,1,1,0,0,0,2,2,0].subtract(-jnp.exp(logc_val+logc_val)),0.)

def test_compute_extending_beginning_motif_reaction_logc_rates():
    motiflength = 4
    number_of_letters = 4
    pseudo_count_concentration = 1.e-12
    logc_val = -3.
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    e4pr = kdi.infer._initialize_empty_fourmer_production_rates(number_of_letters)
    log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))
    log_reaction_rate_constants = log_reaction_rate_constants.at[0,0,0,0,0,1,1,0].set(0.)
    lrr = kdi.infer._compute_extending_beginning_motif_reaction_logc_rates(fourmer_logc,log_reaction_rate_constants)
    assert jnp.allclose(lrr.at[0,0,1,1,0,0,0,2,2,0].subtract(-jnp.exp(logc_val+logc_val)),0.)

def test_compute_produced_motif_reaction_rates():
    motiflength = 4
    number_of_letters = 4
    pseudo_count_concentration = 1.e-12
    logc_val = -3.
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    e4pr = kdi.infer._initialize_empty_fourmer_production_rates(number_of_letters)
    log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))
    log_reaction_rate_constants = log_reaction_rate_constants.at[0,0,0,0,0,1,1,0].set(0.)
    for product_index in [0,1,2,]:
        lrr = kdi.infer._compute_produced_motif_reaction_logc_rates(
                fourmer_logc,
                log_reaction_rate_constants,
                product_index = product_index
                )
        if product_index==1:
            lrr = lrr.at[0,0,1,1,0,0,0,2,2,0].subtract(jnp.exp(logc_val+logc_val))
        assert jnp.allclose(lrr,0.)


test_extension_modes = [
        (
            [(0,0,0,0,0,1,1,0)],
            [(0,1,0,0),(0,1,0,0)],
            [(0,1,1,0)]
            ),
        (
            [(0,0,1,2,1,0,1,0)],
            [(0,1,0,0),(0,2,2,0),(0,1,0,0),(0,2,2,1),(0,1,0,0),(0,2,2,2),(0,1,0,0),(0,2,2,3),(0,1,0,0),(0,2,2,4)],
            [(0,1,2,2),(1,2,2,0),(0,1,2,2),(1,2,2,1),(0,1,2,2),(1,2,2,2),(0,1,2,2),(1,2,2,3),(0,1,2,2),(1,2,2,4)]
            ),
        (
            [(0,0,1,3,4,0,1,0)],
            [(0,1,0,0),(0,2,3,0),(0,1,0,0),(0,2,3,1),(0,1,0,0),(0,2,3,2),(0,1,0,0),(0,2,3,3),(0,1,0,0),(0,2,3,4)],
            [(0,1,2,3),(1,2,3,0),(0,1,2,3),(1,2,3,1),(0,1,2,3),(1,2,3,2),(0,1,2,3),(1,2,3,3),(0,1,2,3),(1,2,3,4)]
            ),
        (
            [(3,1,0,0,0,1,0,4)],
            [(0,3,2,0),(0,1,0,0),(1,3,2,0),(0,1,0,0),(2,3,2,0),(0,1,0,0),(3,3,2,0),(0,1,0,0),(4,3,2,0),(0,1,0,0)],
            [(0,3,2,1),(3,2,1,0),(1,3,2,1),(3,2,1,0),(2,3,2,1),(3,2,1,0),(3,3,2,1),(3,2,1,0),(4,3,2,1),(3,2,1,0),]
            ),
        (
            [(4,2,1,1,2,0,3,3)],
            [(0,4,3,0),(1,4,3,0),(2,4,3,0),(3,4,3,0),(4,4,3,0),
             (0,2,1,0),(0,2,1,0),(0,2,1,0),(0,2,1,0),(0,2,1,0),
             #
             (0,4,3,0),(1,4,3,0),(2,4,3,0),(3,4,3,0),(4,4,3,0),
             (0,2,1,1),(0,2,1,1),(0,2,1,1),(0,2,1,1),(0,2,1,1),
             #
             (0,4,3,0),(1,4,3,0),(2,4,3,0),(3,4,3,0),(4,4,3,0),
             (0,2,1,2),(0,2,1,2),(0,2,1,2),(0,2,1,2),(0,2,1,2),
             #
             (0,4,3,0),(1,4,3,0),(2,4,3,0),(3,4,3,0),(4,4,3,0),
             (0,2,1,3),(0,2,1,3),(0,2,1,3),(0,2,1,3),(0,2,1,3),
             #
             (0,4,3,0),(1,4,3,0),(2,4,3,0),(3,4,3,0),(4,4,3,0),
             (0,2,1,4),(0,2,1,4),(0,2,1,4),(0,2,1,4),(0,2,1,4),
             ],
            [
                (4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),
                (0,4,3,2),(1,4,3,2),(2,4,3,2),(3,4,3,2),(4,4,3,2),
                (3,2,1,0),(3,2,1,0),(3,2,1,0),(3,2,1,0),(3,2,1,0),
                #
                (4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),
                (0,4,3,2),(1,4,3,2),(2,4,3,2),(3,4,3,2),(4,4,3,2),
                (3,2,1,1),(3,2,1,1),(3,2,1,1),(3,2,1,1),(3,2,1,1),
                #
                (4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),
                (0,4,3,2),(1,4,3,2),(2,4,3,2),(3,4,3,2),(4,4,3,2),
                (3,2,1,2),(3,2,1,2),(3,2,1,2),(3,2,1,2),(3,2,1,2),
                #
                (4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),
                (0,4,3,2),(1,4,3,2),(2,4,3,2),(3,4,3,2),(4,4,3,2),
                (3,2,1,3),(3,2,1,3),(3,2,1,3),(3,2,1,3),(3,2,1,3),
                #
                (4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),(4,3,2,1),
                (0,4,3,2),(1,4,3,2),(2,4,3,2),(3,4,3,2),(4,4,3,2),
                (3,2,1,4),(3,2,1,4),(3,2,1,4),(3,2,1,4),(3,2,1,4),
                ]
            ),
        ]
@pytest.mark.parametrize('reaction_channels, reactants, products', test_extension_modes)
def test_compute_total_extension_rates(reaction_channels,reactants,products):
    motiflength = 4
    alphabet = ['a','b','c','d']
    pseudo_count_concentration = 1.e-12#45 # rate constants zero shift
    motif_pseudocounts = 1.e-12
    logc_val = -6.
    number_of_letters = len(alphabet)
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,motif_pseudocounts)
    e4pr = kdi.infer._initialize_empty_fourmer_production_rates(number_of_letters)
    shp = (number_of_letters+1,number_of_letters,)
    shp += shp[::-1]
    shp += shp
    log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones(shp)
    for reaction_channel in reaction_channels:
        log_reaction_rate_constants = log_reaction_rate_constants.at[reaction_channel].set(0.)
    extension_rates = kdi.infer.compute_total_extension_rates(
            fourmer_logc,
            log_reaction_rate_constants,
            number_of_letters = number_of_letters,
            conserve_mass = False)
    er = deepcopy(extension_rates)
    for reactant in  reactants:
        er = er.at[reactant].subtract(-jnp.exp(logc_val+logc_val))
    for product in products:
        er = er.at[product].subtract(jnp.exp(logc_val+logc_val))
    try:
        assert jnp.allclose(er,0.)
    except AssertionError:
        print(f"{jnp.exp(logc_val+logc_val) = }")
        print(f"{jnp.max(er[er!=0.]) = }")
        print(f"{jnp.min(er[er!=0.]) = }")
        raise AssertionError(f"{jnp.asarray(jnp.where(er!=0.)).T = }, {er[er!=0.] = }")
    
    logc_diff = kdi.infer.compute_total_extension_rates(
            fourmer_logc,
            log_reaction_rate_constants,
            number_of_letters = number_of_letters,
            conserve_mass = False)
    c_diff = jnp.exp(fourmer_logc)*logc_diff
    c_diff_dct = kdi._array_to_motif_vector_dct(c_diff[:,1:],motiflength,alphabet)
    c_diff_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')(c_diff_dct)
    try:
        assert jnp.allclose(total_mass(c_diff),0.)
    except AssertionError:
        print(f"total mass diff: {jnp.sum(c_diff[:,1:,:,:])+jnp.sum(c_diff[:,1:,1:,0])}")
        raise AssertionError(f"{jnp.asarray(jnp.where(c_diff!=0.)).T = },\n{c_diff[c_diff!=0.] = }")

def total_mass(c):
    return jnp.sum(c[0,1:,:,0]) + jnp.sum(c[:,1:,1:,1:])+ 2*jnp.sum(c[1:,1:,1:,0]) + jnp.sum(c[0,1:,1:,0])

def test_mass_conservation():
    motiflength = 4
    number_of_letters = 4
    pseudo_count_concentration = 1.e-12
    logc_val = -12
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
    e4pr = kdi.infer._initialize_empty_fourmer_production_rates(number_of_letters)
    log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))
    for tt1,tt4,l1,r4 in iterprod([0,1],repeat=4):
        log_reaction_rate_constants = log_reaction_rate_constants.at[l1*1,0,0,r4*1,tt1*2,1,1,tt4*2].set(0.)
        logc_diff = kdi.infer.compute_total_extension_rates(fourmer_logc, log_reaction_rate_constants)
        c_diff = jnp.exp(fourmer_logc)*logc_diff
        assert jnp.allclose(total_mass(c_diff),0.)
        log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))

    def t(lindices,rindices,tindices):
        fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
        fourmer_logc = fourmer_logc.at[lindices+(0,)].set(-2)
        fourmer_logc = fourmer_logc.at[(0,)+rindices].set(-3)
        fourmer_logc = fourmer_logc.at[tindices].set(-5)
        fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
        log_reaction_rate_constants = jnp.log(pseudo_count_concentration)*jnp.ones((5,4,4,5,5,4,4,5))
        lindices = (lindices[1],lindices[2]-1)
        rindices = (rindices[0]-1,rindices[1])
        tindices = (tindices[0],tindices[1]-1,tindices[2]-1,tindices[3])
        log_reaction_rate_constants = log_reaction_rate_constants.at[lindices+rindices+tindices].set(0.)
        logc_diff = kdi.infer.compute_total_extension_rates(
                fourmer_logc,
                log_reaction_rate_constants,
                soft_reactant_threshold=1.1*jnp.exp(logc_val),
                hard_reactant_threshold=jnp.exp(logc_val),
                )
        c_diff = jnp.exp(fourmer_logc)*logc_diff
        assert jnp.allclose(total_mass(c_diff),0.,atol=pseudo_count_concentration)
    lindices_list = [(0,1,2),(2,1,2)]
    rindices_list = [(2,2,0),(2,2,2)]
    tindices_list = [(0,1,1,0),(0,1,1,1),(1,1,1,0),(1,1,1,1)]
    for lindices, rindices, tindices in iterprod(lindices_list,rindices_list,tindices_list):
        t(lindices,rindices,tindices)
