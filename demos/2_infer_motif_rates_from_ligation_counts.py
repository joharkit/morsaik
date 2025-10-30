import morsaik as kdi
import nifty8.re as jft
from functools import partial
import jax.numpy as jnp
import jax

from os import makedirs
from os.path import exists, isdir

import matplotlib.pyplot as plt

class MotifProductionRatesModel(jft.Model):
    def __init__(
            self,
            motiflength,
            alphabet,
            motif_logconcentrations_trajectory,
            motif_concentration_trajectory_times_array,
            motif_production_rate_constants_prior_mean,
            motif_production_rate_constants_prior_std,
            ):
        if motiflength != 4:
            raise NotImplementedError
        nol = len(alphabet)
        shape = ((nol+1)*nol*nol*(nol+1)*(nol+1)*nol*nol*(nol+1),)
        self.exposure = kdi.infer.collisions_from_motif_concentration_trajectory_array_and_collision_rate_constants_array(
               motif_logconcentrations_trajectory,
               motif_concentration_trajectory_times_array
                )
        self.motif_production_rate_constants = jft.LogNormalPrior(
                motif_production_rate_constants_prior_mean,
                motif_production_rate_constants_prior_std,
                name='log_motif_production_rate_constants',
                shape=shape)
        super().__init__(domain = jft.Vector({'log_motif_production_rate_constants': jax.ShapeDtypeStruct(shape,float)}))
    def __call__(self,xi):
        motif_production_rates_array = self.motif_production_rate_constants(xi).flatten()*self.exposure
        return motif_production_rates_array.reshape(-1)

if __name__=='__main__':
    motiflength = 4
    strand_trajectory_id = '9999_99_99__99_99_99'
    param_file_no = 0
    alphabet = kdi.get.alphabet(strand_trajectory_id)
    plotting_alphabet = ['X','Y']
    complements = [1,0]
    plotformats = ['.pdf']
    motif_production_rate_constants_prior_mean = 1.e-6
    motif_production_rate_constants_prior_std = 1.e-6

    #KL minimization specs
    kl_n_total_iterations = 6
    n_prior_samples = 3
    n_posterior_samples = 3
    rate_equation_motif_trajectory_ensemble_plotpath = f'./Plots/{strand_trajectory_id}-{param_file_no}/prior_'+'{:.0e}'.format(motif_production_rate_constants_prior_mean)+'_{:.0e}'.format(motif_production_rate_constants_prior_std) + "/"
    makedirs(rate_equation_motif_trajectory_ensemble_plotpath, exist_ok=True)
    rate_equation_motif_trajectory_ensemble_archive_path = f'./archive/{strand_trajectory_id}-{param_file_no}/prior_'+'{:.0e}'.format(motif_production_rate_constants_prior_mean)+'_{:.0e}'.format(motif_production_rate_constants_prior_std) + "/"
    makedirs(rate_equation_motif_trajectory_ensemble_archive_path, exist_ok=True)
    optimize_kl_odir = rate_equation_motif_trajectory_ensemble_archive_path + 'optimize_kl/'
    resume_to_kl_odir = isdir(optimize_kl_odir) #set to False if you want to optimize kl again
    makedirs(optimize_kl_odir, exist_ok=True)
    print(f'{motiflength=}')
    print(f'{strand_trajectory_id=}')
    print(f'{param_file_no=}')
    print("plotpath:")
    print(rate_equation_motif_trajectory_ensemble_plotpath)
    print("archive path:")
    print(rate_equation_motif_trajectory_ensemble_archive_path)
    print('get strand reactor data')
    strand_reactor_motifs_trajectory = kdi.get.strand_motifs_trajectory(motiflength, strand_trajectory_id, param_file_no=param_file_no)
    strand_reactor_motifs_trajectory_ensemble = kdi.MotifTrajectoryEnsemble([strand_reactor_motifs_trajectory])
    strand_reactor_parameters = kdi.get.strand_reactor_parameters(strand_trajectory_id, param_file_no=param_file_no)
    strand_reactor_motifs_concentrations_trajectory = kdi.infer.motif_concentration_trajectory_from_motif_number_trajectory(strand_reactor_motifs_trajectory, strand_reactor_parameters['c_ref'])
    strand_reactor_motifs_concentrations_trajectory_ensemble = kdi.MotifTrajectoryEnsemble([strand_reactor_motifs_concentrations_trajectory])
    print('load ligation counts and strand number trajectories')
    strand_reactor_motif_production_counts = []
    total_strand_reactor_motif_production_counts = jnp.zeros((36,36), dtype=int)
    strand_motifs_productions_trajectory_ensemble = kdi.get.strand_motifs_productions_trajectory_ensemble(motiflength, strand_trajectory_id, param_file_no=param_file_no)
    for simulations_run_no in range(len(strand_motifs_productions_trajectory_ensemble.trajectories)):
        strand_motifs_productions_trajectory_array, strand_motifs_productions_trajectory_times_array = kdi._motif_production_trajectory_as_array(strand_motifs_productions_trajectory_ensemble.trajectories[simulations_run_no])
        strand_reactor_motif_production_counts += [jnp.sum(strand_motifs_productions_trajectory_array, axis = 0, dtype = int).reshape(-1)]
        total_strand_reactor_motif_production_counts = total_strand_reactor_motif_production_counts + jnp.asarray(strand_reactor_motif_production_counts[simulations_run_no]).reshape((36,36))
    kdi.plot.motif_production_rates(
           total_strand_reactor_motif_production_counts,
           plotting_alphabet,
           plotpath = rate_equation_motif_trajectory_ensemble_plotpath + f'ligation_counts-',
           plotformats = plotformats,
           )
    strand_reactor_motif_concentration_trajectory_array, strand_reactor_motif_concentration_trajectory_times_array = kdi._motif_trajectory_as_array(strand_reactor_motifs_concentrations_trajectory)
    strand_reactor_motif_logconcentrations_trajectory = jnp.log(strand_reactor_motif_concentration_trajectory_array + kdi.utils.unzero(strand_reactor_motif_concentration_trajectory_array))

    # strand_reactor_motif_production_counts = [jnp.array([1,2],dtype=int),jnp.array([0,2],dtype=int)]
    print('setup MotifProductionRatesModel')

    prng_key = jax.random.PRNGKey(32)
    prng_key, kl_key, sample_key = jax.random.split(prng_key,3)
    strand_reactor_motif_logconcentrations_trajectory = jnp.log(strand_reactor_motif_concentration_trajectory_array + kdi.utils.unzero(strand_reactor_motif_concentration_trajectory_array))
    motif_production_rates_model = MotifProductionRatesModel(
            motiflength,
            alphabet,
            strand_reactor_motif_logconcentrations_trajectory,
            strand_reactor_motif_concentration_trajectory_times_array,
            motif_production_rate_constants_prior_mean,
            motif_production_rate_constants_prior_std,
            )

    print('plot prior motif_production_rates')
    for sample_number in range(n_prior_samples):
        sample_key, prior_sample_key = jax.random.split(prng_key,2)
        prior_latent_sample = jft.random_like(prior_sample_key, motif_production_rates_model.domain)
        kdi.plot.motif_production_rates(
               motif_production_rates_model(prior_latent_sample).reshape((36,36)),
               plotting_alphabet,
               plotpath = rate_equation_motif_trajectory_ensemble_plotpath + f'{sample_number}-prior-',
               plotformats = plotformats,
               )

    motif_production_rates_estimate = jft.Vector({'log_motif_production_rate_constants' : jnp.ones(motif_production_rates_model.domain.shape, dtype=float)})
    minimizer_func = partial(jft.optimize_kl,key = kl_key,
                             n_total_iterations = kl_n_total_iterations,
                             n_samples = n_posterior_samples,
                             #resume = resume_to_kl_odir,
                             odir = optimize_kl_odir)
    print('infer motif_production_rates_array_from_motif_production_counts')
    log_motif_production_rates_posterior_samples, _ = kdi.infer.motif_production_rates_array_from_motif_production_counts(
            motif_production_rates_model,
            motif_production_rates_estimate,
            strand_reactor_motif_production_counts,
            sample_key,
            minimizer_func
            )

    print('plot motif_production_rates')
    sample_number = 0
    for log_motif_production_rates_posterior_sample in log_motif_production_rates_posterior_samples:
       kdi.plot.motif_production_rates(
               motif_production_rates_model(log_motif_production_rates_posterior_sample).reshape((36,36)),
               plotting_alphabet,
               plotpath = rate_equation_motif_trajectory_ensemble_plotpath + f'{sample_number}-',
               plotformats = plotformats,
               )
       sample_number += 1

    print('plot posterior motif production rate constants inferred from strand reactor parameters')
    sample_number = 0
    motif_production_rate_constants_model = motif_production_rates_model.motif_production_rate_constants
    for log_motif_production_rates_posterior_sample in log_motif_production_rates_posterior_samples:
        motif_production_rate_constants = motif_production_rate_constants_model(log_motif_production_rates_posterior_sample)
        kdi.plot.motif_production_rates(
                motif_production_rate_constants.reshape((36,36)),['A','T'],
                plotpath = rate_equation_motif_trajectory_ensemble_plotpath + f'mprc_{sample_number}-',
                plotformats = plotformats,
                )
        sample_number += 1
