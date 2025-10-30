import morsaik as kdi
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs

from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
import jax.numpy as jnp

def infer_motif_trajectory_from_dimerization_rate_constant(
        motiflength : int = 4,
        alphabet : list = ['A'],
        unit : kdi.Unit = kdi.make_unit('mol')/kdi.make_unit('L'),
        motif_production_rate_constant = 1.,
        initial_monomer_concentration : float = 1.e-2,
        total_mass : float = 1.e-2+2*1.e-5
    ) -> kdi.MotifTrajectory:
    f0 = initial_monomer_concentration/total_mass
    monomer_concentrations = initial_monomer_concentration*np.arange(100)/100
    def t(motif_concentration,
          total_mass,
          motif_production_rate_constant,
          integration_constant : float = 0.
          ):
        f = motif_concentration/total_mass
        #return integration_constant-motif_production_rate_constant*(1+f*np.log(1/f-1))/(f*total_mass**2)
        #return integration_constant + motif_production_rate_constant*(f*np.log(1-1/f) +1)/(f*total_mass**2)
        return (1/f + np.log(1-f) - np.log(f))/(motif_production_rate_constant*total_mass**2) + integration_constant
    integration_constant = -t(initial_monomer_concentration, total_mass, motif_production_rate_constant, integration_constant = 0.)
    times = kdi.TimesVector(
        t(monomer_concentrations, total_mass, motif_production_rate_constant, integration_constant = integration_constant),
        units = kdi.read.symbol_config('time', unitformat=True)
    )
    motif_concentrations = np.zeros(times.domain.shape+(2,1,2,2))
    motif_concentrations[:,0,0,0,0] = monomer_concentrations
    motif_concentrations[:,0,0,1,0] = 0.5*(total_mass-monomer_concentrations)
    motif_vectors = [[]]*monomer_concentrations.shape[0]
    for times_index in  range(monomer_concentrations.shape[0]):
        mvd = kdi._array_to_motif_vector_dct(
            motif_concentrations[times_index],
            motiflength,
            alphabet,
        )
        motif_vectors[times_index] = kdi.MotifVector(motiflength, alphabet, unit)(mvd)
    return kdi.MotifTrajectory(motif_vectors, times)

def simulate_monomer_dimer_system(
        motiflength : int = 4,
        alphabet : list = ['A'],
        unit : kdi.Unit = kdi.make_unit('mol')/kdi.make_unit('L'),
        motif_production_rate_constant : int = 1.,
        maximum_ligation_window_length : int = 4,
        time_unit : kdi.Unit = kdi.read.symbol_config('time', unitformat=True),
        ode_integration_method = 'LSODA'
    ) -> kdi.MotifTrajectory:
    resolution_factor = 1e-1 #16*16 #1e-9
    t_span = (0,1.e6)#(0,1.e12)
    complements = [0,]
    motif_production_array = np.zeros((2,1,1,2)*2)
    motif_production_array[0,0,0,0,0,0,0,0] = 1.
    motif_production_rate_constants = kdi._array_to_motif_production_vector(
        motif_production_array,
        motiflength,
        alphabet,
        kdi.make_unit('L')**2/kdi.make_unit('mol')**2/time_unit,
        maximum_ligation_window_length
    )
    breakage_rate_constants = kdi.MotifBreakageVector(motiflength,alphabet,1./time_unit)
    breakage_vector_dct = kdi._create_empty_motif_breakage_dct(motiflength, alphabet)
    breakage_vector_array = np.zeros((2,1,1,2))
    breakage_vector_dct = kdi._array_to_motif_breakage_vector(
        breakage_vector_array,
        motiflength,
        alphabet,
        1./time_unit
    )
    breakage_rates = breakage_rate_constants(breakage_vector_dct)
    motif_vector_array = np.zeros((2,1,2,2))
    motif_vector_array[0,0,0,0] = 1.e-2
    motif_vector_array[0,0,1,0] = 1.e-5
    initial_concentrations_dct = kdi._array_to_motif_vector_dct(
        motif_vector_array,
        motiflength,
        alphabet
    )
    initial_motif_concentrations_vector = kdi.MotifVector(motiflength,alphabet,'mol/L')
    initial_motif_concentrations_vector = initial_motif_concentrations_vector(initial_concentrations_dct)
    times = kdi.TimesVector(1./resolution_factor*np.arange(*(resolution_factor*t_span[0], resolution_factor*t_span[1]), dtype = np.float64), time_unit)

    motif_production_log_rate_constants = kdi._array_to_motif_production_vector(
        jnp.zeros(kdi._motif_production_array_shape(len(alphabet),motiflength)),
        motiflength,
        alphabet,
        kdi.make_unit(1),
        maximum_ligation_window_length
    )

    return kdi.infer.fourmer_trajectory_from_rate_constants(
        motif_production_rate_constants,
        motif_production_log_rate_constants,
        breakage_rates,
        initial_motif_concentrations_vector,
        times=times,
        complements=complements,
        ode_integration_method = ode_integration_method,# 'BDF',#'Radau'
    )

def simulate_monomer_dimer_system_with_jax():
    def produce_dimers(t,y, production_rate_constant):
        return jnp.array([-2*production_rate_constant*y[0]**2*y[1],production_rate_constant*y[0]**2*y[1],])
    def break_dimers(t,y,breakage_rate_constant):
        return jnp.array([+2*breakage_rate_constant*y[0]**2*y[1],-breakage_rate_constant*y[0]**2*y[1],])
    def f_monomer_dimer_system(t, y, args):
        return produce_dimers(t,y,args[0]) + break_dimers(t,y,args[1])
    production_rate_constant = 0.1
    breakage_rate_constant = 0.01
    args = [production_rate_constant, breakage_rate_constant]
    term = ODETerm(f_monomer_dimer_system)
    solver = Dopri5()
    y0 = jnp.array([2., 3.])
    solution = diffeqsolve(term, solver, t0=0, t1=10, dt0=0.1, args=args, y0=y0, saveat = SaveAt(dense=True))
    return solution.evaluate(1.)

if __name__=='__main__':
    scenario = 'monomer-dimer-system'
    ode_integration_method = 'LSODA'
    analytical_motif_trajectory = infer_motif_trajectory_from_dimerization_rate_constant()
    simulated_motif_trajectory = simulate_monomer_dimer_system()

    plotpath = './Plots/'+scenario+'/'
    makedirs(plotpath, exist_ok=True)

    td_plot_parameters = {
            'linestyle' : '-',
            'color' : kdi.plot.standard_colorbar()(1.),
            'alpha' : 1.,
            'label' : 'Theory',
            }
    sd_plot_parameters = {
            'linestyle' : '-.',
            'color' : kdi.plot.standard_colorbar()(0.5),
            'alpha' : 0.3,
            'label' : 'Strand',
            }
    md_plot_parameters = {
            'linestyle' : '--',
            'color' : kdi.plot.standard_colorbar()(0.),
            'alpha' : 1.,
            'label' : 'Motif',
            }

    plt.plot(analytical_motif_trajectory.times.val, analytical_motif_trajectory.motifs['length1strand'].val[:,0], **td_plot_parameters)
    plt.plot(analytical_motif_trajectory.times.val, analytical_motif_trajectory.motifs['length2strand'].val[:,0,0], **td_plot_parameters)
    plt.plot(simulated_motif_trajectory.times.val, simulated_motif_trajectory.motifs['length1strand'].val[:,0], **md_plot_parameters)
    plt.plot(simulated_motif_trajectory.times.val, simulated_motif_trajectory.motifs['length2strand'].val[:,0,0], **md_plot_parameters)
    plt.xlabel(f"Time [{kdi.transform_unit_to_str(kdi.read.symbol_config('time', unitformat=True))}]")
    plt.ylabel('Concentration [{}]'.format(simulated_motif_trajectory.unit))
    plt.xlim((0,0.5e6))
    plt.ylim((0,0.01))
    for fileformat in ['pdf']:
        figname = plotpath+f'concentrations_{ode_integration_method}.{fileformat}'
        plt.savefig(figname)
        print(f'saved {figname}')
    plt.close()

    kdi.plot.motif_entropy(kdi.MotifTrajectoryEnsemble([analytical_motif_trajectory]), **td_plot_parameters)
    kdi.plot.motif_entropy(kdi.MotifTrajectoryEnsemble([simulated_motif_trajectory]), **md_plot_parameters)
    plt.xscale('log')
    plt.yscale('log')
    for fileformat in ['pdf']:
        figname = plotpath+f'entropy_{ode_integration_method}.{fileformat}'
        plt.savefig(figname)
        print(f'saved {figname}')
    plt.close()
