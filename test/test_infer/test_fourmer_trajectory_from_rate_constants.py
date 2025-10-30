import jax.numpy as jnp
import itertools
import pytest
from copy import deepcopy

import morsaik as kdi

def setup_c(test_motifs=[(0,1,0,0),(0,2,3,0),(0,1,3,2),(1,3,2,4),(3,2,4,0)],
            test_concentration=0.1,
            nol=4,
            motiflength = 4):
    c = jnp.zeros([nol+1,]*motiflength)
    for motif in test_motifs:
        c = c.at[motif].set(test_concentration)
    return c

def unzero(c,shift,shiftmode='abs'):
    if shiftmode=='abs':
        return c.at[c<shift].set(shift)
    else:
        raise NotImplementedError(f"{shiftmode=}")

def test_monotonous_trajectory():
    motiflength = 4
    alphabet = ['a','b']
    complements = [1,0]
    unit = kdi.make_unit('')
    maximum_ligation_window_length = 4

    empv = kdi._create_empty_motif_production_dict(motiflength,
            alphabet,
            maximum_ligation_window_length
            )
    motif_production_vector = kdi.MotifProductionVector(motiflength,alphabet,unit, maximum_ligation_window_length)
    motif_production_vector = motif_production_vector(empv)

    units = [kdi.make_unit(''),1./kdi.make_unit('s')]
    motif_breakage_vector_dct = kdi._create_empty_motif_breakage_dct(
            motiflength,
            alphabet
            )
    motif_breakage_vector = kdi.MotifBreakageVector(motiflength,alphabet,unit)
    motif_breakage_vector = motif_breakage_vector(motif_breakage_vector_dct)

    motif_concentration_dct = kdi._create_empty_motif_vector_dct(motiflength,alphabet=alphabet)

    motif_concentration_vector = kdi.MotifConcentrationVector(motiflength,alphabet)
    motif_concentration_vector = motif_concentration_vector(motif_concentration_dct)

    times_vector = jnp.asarray([0.,1.])
    times_vector = kdi.TimesVector(times_vector,kdi.make_unit('s'))

    actual_fourmer_trajectory = kdi.infer.fourmer_trajectory_from_rate_constants(
            motif_production_rate_constants = motif_production_vector,
            motif_production_log_rate_constants = motif_production_vector,
            breakage_rate_constants = motif_breakage_vector,
            initial_motif_concentrations_vector = motif_concentration_vector,
            times = times_vector,
            complements = complements,
            )
    assert jnp.allclose(
            kdi._motif_trajectory_as_array(actual_fourmer_trajectory)[0],
            kdi._motif_vector_as_array(motif_concentration_vector)[None]
            )
test_extension_modes = [
        (
            [(0,0,0,0,0,1,1,0)],
            [(0,1,0,0),(0,1,0,0)],
            [(0,1,1,0)]
            ),
        ]
@pytest.mark.parametrize('reaction_channels, reactants, products', test_extension_modes)
def test_compute_total_extension_rates(reaction_channels,reactants,products):
    motiflength = 4
    alphabet = ['a','b','c','d']
    pseudo_count_concentration = 1.e-12 # rate constants zero shift
    motif_pseudocounts = 1.e-12
    logc_val = -6.
    number_of_letters = len(alphabet)
    fourmer_logc = logc_val*jnp.ones((number_of_letters+1,)*motiflength)
    fourmer_logc = kdi.infer._set_invalid_logc_to_log0(fourmer_logc,pseudo_count_concentration)
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
        assert jnp.allclose(kdi.infer.total_mass(c_diff_vector),0.)
    except AssertionError:
        print(kdi.infer.total_mass(c_diff_vector))
        print(c_diff.shape)
        print(f"total mass diff: {jnp.sum(c_diff[:,1:,:,:])+jnp.sum(c_diff[:,1:,1:,0])}")
        raise AssertionError(f"{jnp.asarray(jnp.where(c_diff!=0.)).T = },\n{c_diff[c_diff!=0.] = }")
    assert jnp.allclose(kdi.infer.total_mass(c_diff_vector),0.)


test_extension_modes = [
        (
            # (0,1,0,0) -> (0,2,0,0)
            # (0,2,3,0) -> (0,4,1,0)
            # (0,1,3,2,4,0) -> (0,3,1,4,2,0)
            #4,1,2,0
            #continuing_motif =(1,3,2,4)
            [(0,1,0,0)],
            [(0,2,3,0)],
            [(4,1,2,0)],
            [(0,1,2,3), (1,2,3,0)],
            ),
        ]
@pytest.mark.parametrize('left_reactants, right_reactants, templates, products', test_extension_modes)
def test_fourmer_production_equations(left_reactants, right_reactants, templates, products):
    t_eval = jnp.linspace(0.,1.e-1,2)
    mprc_value = 1.
    breakage_rate_constants = 0.
    delta_t = t_eval[-1]-t_eval[0]
    nol = 4
    motiflength = 4
    complements = [1,0,3,2]
    complements_0 = jnp.concatenate([jnp.zeros(1),jnp.asarray(complements)+1],dtype=int)
    # motif production rate constant
    mprc = jnp.zeros([nol+1,]*(2*motiflength))
    for ii,jj,kk,ll in itertools.product(range(nol+1),repeat=4):
        ci = complements_0[ii]
        cj = complements_0[jj]
        ck = complements_0[kk]
        cl = complements_0[ll]
        if jj*kk>0:
            mprc = mprc.at[ii,jj,kk,ll,cl,ck,cj,ci].set(mprc_value)
    alpha = 0.

    pseudo_count_concentration = 1.e-12
    test_concentration = 0.1
    test_motifs = left_reactants + right_reactants + templates + products

    c = setup_c(
            test_motifs=test_motifs,
            test_concentration=test_concentration,
            nol=nol,
            motiflength=motiflength
            )

    logc_old = jnp.log(unzero(c, shift=pseudo_count_concentration, shiftmode='abs'))
    logc_old = kdi.infer._set_invalid_logc_to_log0(logc_old,pseudo_count_concentration=pseudo_count_concentration)

    args = {
            'influx_rate_constants' : alpha,
            'fourmer_production_log_rate_constants': jnp.zeros(mprc.shape),
            'fourmer_production_rate_constants' : mprc,
            'breakage_rate_constants' : breakage_rate_constants,
            'complements' : jnp.asarray(complements),
            'motiflength' : motiflength
            }
    #Delta t = 1
    logc_ptf = logc_old.reshape(-1) + delta_t*kdi.infer._fourmer_production_equations(1., logc_old.reshape(-1), args)
    logc_ptf = logc_ptf.reshape(logc_old.shape)

    # possible productions:
    # (0,1,0,0) + (0,1,0,0) --(0,2,2,0)--> 0
    # (0,1,0,0) + (0,2,3,0) --(0,4,1;2,0)--> (0,1,2,3) + (1,2,3,0)
    # (0,1,0,0) + (0,4,1,2,0) --(0,1,2,3,2,0) --> 0
    # (0,2,3,0) + (0,1,0,0) --(0,2,4,1,0)--> 0
    # (0,2,3,0) + (0,2,3,0) --(0,4,1;4,1,0)--> 0
    # (0,2,3,0) + (0,4,1,2,0)) --(0,1,2,3;4,1,0)--> 0
    # (0,4,1,2,0) + (0,1,0,0) --(0,2;1,2,3,0)--> 0
    # (0,4,1,2,0) + (0,2,3,0) --(0,4,1;1,2,3,0)--> 0
    # (0,4,1,2,0) + (0,4,1,2,0) --(0,1,2,3;1,2,3--> 0
    # initialize logc_new
    logc_new = deepcopy(logc_old)
    # consume left reactant
    for left_reactant in left_reactants:
        logc_new = logc_new.at[left_reactant].subtract(mprc_value*test_concentration*test_concentration*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)
    # consume right reactant
    for right_reactant in right_reactants:
        logc_new = logc_new.at[right_reactant].subtract(mprc_value*test_concentration*test_concentration*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)
    # produce emerging motif
    for produced_motif in products:
        logc_new = logc_new.at[produced_motif].add(mprc_value*test_concentration**3*jnp.exp(-logc_old[produced_motif])*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)

    try:
        assert jnp.allclose(
                jnp.exp(logc_ptf),
                jnp.exp(logc_new),#desired
                rtol=1.e-4,
                atol=1.e-10
                )
    except AssertionError:
        indicess = jnp.abs(logc_ptf-logc_new)>1.e-10+1.e-4*jnp.abs(logc_new)
        raise AssertionError(f"At {jnp.asarray(jnp.where(indicess)).T}; logc_ptf: {logc_ptf[indicess]}; logc_new: {logc_new[indicess]}")
    # check total mass
    try:
        assert jnp.allclose(
            jnp.sum(jnp.exp(logc_old)[:,1:,:,:].flatten()) + jnp.sum(jnp.exp(logc_old)[:,:,1:,0].flatten()),
            jnp.sum(jnp.exp(logc_new)[:,1:,:,:].flatten()) + jnp.sum(jnp.exp(logc_new)[:,:,1:,0].flatten()),
            rtol=1.e-4,
            atol=1.e-10
            )
    except AssertionError:
        raise AssertionError("old total mass:"+
        f"{jnp.sum(jnp.exp(logc_old)[:,1:,:,:].flatten()) + jnp.sum(jnp.exp(logc_old)[:,:,1:,0].flatten())}; " +
        f"new total mass: {jnp.sum(jnp.exp(logc_new)[:,1:,:,:].flatten()) + jnp.sum(logc_new[:,:,1:,0].flatten())}"
                             )

def test_breakage_terms_function():
    t_eval = jnp.linspace(0.,1.e-5,2)
    delta_t = t_eval[-1]-t_eval[0]
    nol = 4
    motiflength = 4
    complements = [1,0,3,2]
    complements_0 = jnp.concatenate([jnp.zeros(1),jnp.asarray(complements)+1],dtype=int)
    breakage_rate_constant = 1.e-5
    # motif production rate constant
    mprc = jnp.zeros([nol+1,]*(2*motiflength))
    for ii,jj,kk,ll in itertools.product(range(nol+1),repeat=4):
        ci = complements_0[ii]
        cj = complements_0[jj]
        ck = complements_0[kk]
        cl = complements_0[ll]
        if jj*kk>0:
            mprc = mprc.at[ii,jj,kk,ll,cl,ck,cj,ci].set(1.)
    alpha = 0.

    # (0,1,0,0) -> (0,2,0,0)
    # (0,2,3,0) -> (0,4,1,0)
    # (0,1,3,2,4,0) -> (0,3,1,4,2,0)
    monomer_motif = (0,1,0,0)
    dimer_motif = (0,2,3,0) 
    beginning_motif = (0,1,3,2)
    continuing_motif =(1,3,2,4)
    end_motif =(3,2,4,0)
    pseudo_count_concentration = 1.e-12
    test_concentration = 0.1
    test_motifs = [monomer_motif,dimer_motif,beginning_motif,continuing_motif,end_motif]

    c = setup_c(
            test_motifs=test_motifs,
            test_concentration=test_concentration,
            nol=nol,
            motiflength=motiflength
            )

    breakage_rate_constants = breakage_rate_constant*jnp.ones((nol+1,)*motiflength)
    logc_old = jnp.log(unzero(c, shift=pseudo_count_concentration, shiftmode='abs'))
    logc_old = kdi.infer._set_invalid_logc_to_log0(logc_old,pseudo_count_concentration=pseudo_count_concentration)

    args = {
            'influx_rate_constants' : alpha,
            'fourmer_production_rate_constants' : mprc,
            'breakage_rate_constants' : breakage_rate_constants,
            'complements' : jnp.asarray(complements), 
            'motiflength' : motiflength
            }
    #Delta t = 1
    logc_btf = logc_old.reshape(-1) + delta_t*kdi.infer._fourmer_breakage_equations(1., logc_old.reshape(-1), args)
    logc_btf = logc_btf.reshape(logc_old.shape)

    x_0, times_array, _ = kdi.infer._integrate_motif_rate_equations(
            logc_old,
            number_of_letters=nol,
            motiflength=motiflength,
            complements = jnp.asarray(complements),
            concentrations_are_logarithmized=True,
            fourmer_production_rate_constants = mprc,
            breakage_rate_constants = breakage_rate_constants,
            t_eval = t_eval,
            pseudo_count_concentration=pseudo_count_concentration,
            first_step = 1.e-10,
            ode_integration_method = 'RK23'
            )

    # possible breakage:
    # (0,2,3,) -> (0,2,0,0) + (0,3,0,0)
    # (0,1,3,2,4,0)
    # 1. -> (0,1,0,0) + (0,3,2,4,0)
    # 2. -> (0,1,3,0) + (0,2,4,0)
    # 3. -> (0,1,3,2,0) + (0,4,0,0)
    # initialize logc_new
    logc_new = logc_old
    # break continuations
    logc_new = logc_new.at[1:,1:,1:,1:].subtract(3*breakage_rate_constant*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)
    # break ends
    logc_new = logc_new.at[1:,1:,1:,0].subtract(2*breakage_rate_constant*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)
    # break beginnings
    logc_new = logc_new.at[0,1:,1:,1:].subtract(2*breakage_rate_constant*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)
    # break dimers
    logc_new = logc_new.at[0,1:,1:,0].subtract(1.*breakage_rate_constant*delta_t)
    logc_new = kdi.infer._set_invalid_logc_to_log0(logc_new,pseudo_count_concentration=pseudo_count_concentration)

    # break to ends
    broken_dimer_monomer = dimer_motif[:2]+(0,0)
    broken_beginning_dimer = beginning_motif[:-1]+(0,)
    broken_beginning_monomer = beginning_motif[:-2]+(0,0)
    broken_continuation_end = continuing_motif[:-1]+(0,)
    #
    ideal_logc_diff = jnp.zeros(logc_old.shape)
    ideal_logc_diff = ideal_logc_diff.at[broken_dimer_monomer].add(
            breakage_rate_constant*jnp.exp(logc_old[dimer_motif]-logc_old[broken_dimer_monomer])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_beginning_dimer].add(
            breakage_rate_constant*jnp.exp(logc_old[beginning_motif]-logc_old[broken_beginning_dimer])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_beginning_monomer].add(
            breakage_rate_constant*jnp.exp(logc_old[beginning_motif]-logc_old[broken_beginning_monomer])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_continuation_end].add(
            breakage_rate_constant*jnp.exp(logc_old[continuing_motif]-logc_old[broken_continuation_end])
            )
    #
    logc_new = logc_new + ideal_logc_diff*delta_t

    #break to beginnings
    broken_dimer_monomer = (0,)+dimer_motif[2:]+(0,)
    broken_continuation_beginning = (0,)+continuing_motif[1:]
    broken_end_dimer = (0,)+end_motif[1:]
    broken_end_monomer = (0,)+end_motif[2:]+(0,)
    #
    ideal_logc_diff = jnp.zeros(logc_old.shape)
    ideal_logc_diff = ideal_logc_diff.at[broken_dimer_monomer].add(
            breakage_rate_constant*jnp.exp(logc_old[dimer_motif]-logc_old[broken_dimer_monomer])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_continuation_beginning].add(
            breakage_rate_constant*jnp.exp(logc_old[continuing_motif]-logc_old[broken_continuation_beginning])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_end_dimer].add(
            breakage_rate_constant*jnp.exp(logc_old[end_motif]-logc_old[broken_end_dimer])
            )
    ideal_logc_diff = ideal_logc_diff.at[broken_end_monomer].add(
            breakage_rate_constant*jnp.exp(logc_old[end_motif]-logc_old[broken_end_monomer])
            )
    #
    logc_new = logc_new + ideal_logc_diff*delta_t

    # possible ligations:
    # (3,2,4,1,0,0;0,2,3,0)
    # mprc*left_reactant*right_reactant*template
    ligation_term = logc_old[end_motif]+logc_old[monomer_motif]+logc_old[dimer_motif]
    ligation_term = jnp.exp(ligation_term)
    ligation_term = jnp.multiply(ligation_term,mprc[end_motif[1:-1]+monomer_motif[1:-1]+dimer_motif])
    logc_new = logc_new.at[end_motif].subtract(ligation_term*jnp.exp(-logc_old[end_motif])*delta_t)
    logc_new = logc_new.at[monomer_motif].subtract(ligation_term*jnp.exp(-logc_old[monomer_motif])*delta_t)
    produced_motif = end_motif[:-1]+monomer_motif[1:-2]
    logc_new = logc_new.at[produced_motif].add(ligation_term*jnp.exp(-logc_old[produced_motif])*delta_t)
    produced_motif = end_motif[1:-1]+monomer_motif[1:-1]
    logc_new = logc_new.at[produced_motif].add(ligation_term*jnp.exp(-logc_old[produced_motif])*delta_t)
    # (0,2,3,1,0,0;3,2,4,0)
    ligation_term = jnp.exp(logc_old[monomer_motif]+logc_old[end_motif])
    ligation_term = jnp.multiply(ligation_term,mprc[dimer_motif[1:-1]+monomer_motif[1:-1]+end_motif])
    logc_new = logc_new.at[dimer_motif].subtract(ligation_term*delta_t)

    ligation_term = jnp.exp(logc_old[dimer_motif]+logc_old[end_motif])
    ligation_term = jnp.multiply(ligation_term,mprc[dimer_motif[1:-1]+monomer_motif[1:-1]+end_motif])
    logc_new = logc_new.at[monomer_motif].subtract(ligation_term*delta_t)

    ligation_term = logc_old[dimer_motif]+logc_old[monomer_motif]+logc_old[end_motif]
    ligation_term = jnp.exp(ligation_term)
    ligation_term = jnp.multiply(ligation_term,mprc[dimer_motif[1:-1]+monomer_motif[1:-1]+end_motif])
    produced_motif = dimer_motif[:-1]+monomer_motif[1:-2]
    logc_new = logc_new.at[produced_motif].add(ligation_term*jnp.exp(-logc_old[produced_motif])*delta_t)
    produced_motif = dimer_motif[1:-1]+monomer_motif[1:-1]
    logc_new = logc_new.at[produced_motif].add(ligation_term*jnp.exp(-logc_old[produced_motif])*delta_t)

    x_0 = x_0.reshape((-1,)+logc_new.shape)
    print(jnp.array(jnp.where(jnp.abs(x_0[-1]-logc_new)>1.e-10+1.e-4*jnp.abs(logc_new))))
    print(jnp.array(jnp.where((x_0[-1]-logc_new)>1.e-10+1.e-4*jnp.abs(logc_new))))
    print(jnp.array(jnp.where((logc_new-x_0[-1])>1.e-10+1.e-4*jnp.abs(logc_new))))

    assert jnp.allclose(logc_btf,
            logc_new,#desired
            rtol=1.e-4,
            atol=1.e-10
            )
    # check total mass
    print('old logc')
    print(jnp.sum(logc_old[:,1:,:,:].flatten()) + jnp.sum(logc_old[:,:,1:,0].flatten()))
    print('new logc')
    print(jnp.sum(logc_new[:,1:,:,:].flatten()) + jnp.sum(logc_new[:,:,1:,0].flatten()))
    assert jnp.allclose(
            jnp.sum(jnp.exp(logc_old)[:,1:,:,:].flatten()) + jnp.sum(jnp.exp(logc_old)[:,:,1:,0].flatten()),
            jnp.sum(jnp.exp(logc_new)[:,1:,:,:].flatten()) + jnp.sum(jnp.exp(logc_new)[:,:,1:,0].flatten()),
            rtol=1.e-4,
            atol=1.e-10
            )
