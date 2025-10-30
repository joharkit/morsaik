import morsaik as kdi
import nifty8.re as jft
import jax.numpy as jnp
from functools import partial
import jax

def test_motif_production_rates_array_from_motif_production_counts():
    motif_production_counts = [jnp.array([1,2],dtype=int),jnp.array([0,2],dtype=int)]

    # infer motif production rate constants from ligation counts
    class MotifProductionRatesModel(jft.Model):
        def __init__(self):
            super().__init__(domain = jft.Vector({'xi': jax.ShapeDtypeStruct((2,),float)}))
        def __call__(self,xi):
            return jnp.exp(xi['xi'])

    prng_key = jax.random.PRNGKey(32)
    prng_key, kl_key, sample_key = jax.random.split(prng_key,3)
    motif_production_rates_model = MotifProductionRatesModel()
    motif_production_rates_estimate = jft.Vector({'xi' : jnp.array([1.,1.], dtype=float)})
    minimizer_func = partial(jft.optimize_kl,key = kl_key, n_total_iterations=2,n_samples=2)
    motif_production_rates_samples = kdi.infer.motif_production_rates_array_from_motif_production_counts(
            motif_production_rates_model,
            motif_production_rates_estimate,
            motif_production_counts,
            sample_key,
            minimizer_func
            )
