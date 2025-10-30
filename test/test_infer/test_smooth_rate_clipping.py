import morsaik as kdi
import jax.numpy as jnp
import pytest

def test__clip_smoothly():
    nol,ml=4,4
    hard_reactant_threshold = .5e-1
    soft_reactant_threshold = 1.e-1
    fourmer_c = jnp.zeros((nol+1,)*ml)
    fourmer_c = fourmer_c.at[0,1,0,0].set(hard_reactant_threshold)
    fourmer_c = fourmer_c.at[0,2,0,0].set(soft_reactant_threshold)
    fourmer_c = fourmer_c.at[0,1,1,0].set(2*soft_reactant_threshold)
    fourmer_c = fourmer_c.at[0,1,2,0].set(jnp.mean(jnp.array([hard_reactant_threshold,soft_reactant_threshold])))
    fourmer_c = fourmer_c.at[0,2,1,0].set(hard_reactant_threshold/2.)
    fourmer_c = fourmer_c.at[0,2,2,0].set(soft_reactant_threshold)
    w_excessive = kdi.infer._clip_smoothly(fourmer_c,soft_reactant_threshold=soft_reactant_threshold,hard_reactant_threshold=hard_reactant_threshold)
    w_extensive = kdi.infer._clip_smoothly(fourmer_c[:,1:],soft_reactant_threshold=soft_reactant_threshold,hard_reactant_threshold=hard_reactant_threshold)
    # fourmer_c.at[0,1,0,0].set(hard_reactant_threshold)
    assert jnp.max(w_extensive[0,0,0])==0.
    # fourmer_c.at[0,2,0,0].set(soft_reactant_threshold)
    assert jnp.max(w_extensive[0,0,1])==1.
    # fourmer_c.at[0,1,1,0].set(2*soft_reactant_threshold)
    assert jnp.max(w_extensive[0,1,0])==1.
    # fourmer_c.at[0,1,2,0].set(jnp.mean(jnp.array([hard_reactant_threshold,soft_reactant_threshold])))
    assert jnp.max(w_extensive[0,1,1])<1.
    assert jnp.max(w_extensive[0,1,1])>0.
    # fourmer_c.at[0,2,1,0].set(hard_reactant_threshold/2.)
    assert jnp.max(w_extensive[0,2,0])==0.
    # fourmer_c.at[0,2,2,0].set(soft_reactant_threshold)
    assert jnp.max(w_extensive[0,2,1])==1.
