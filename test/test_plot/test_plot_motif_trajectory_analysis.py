import morsaik as kdi
import matplotlib.pyplot as plt

def test_plot_motif_trajectory_analysis():
    motiflength = 4
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0

    motif_trajectory_ensemble = kdi.get.strand_motifs_trajectory_ensemble(
        motiflength,
        strand_trajectory_id,
        param_file_no=0
    )

    plot_instructions = {
        'plot_motif_entropy' : 1,
        'plot_system_level_motif_zebraness' : 1,
        }
    plot_parameters = {
        'linestyle' : '-',
        'color' : 'b',
        'alpha' : 0.3,
        'label' : 'Strand',
        }

    plt.close('all')
    kdi.plot.motif_trajectory_analysis(
        [motif_trajectory_ensemble],
        plot_instructions,
        [plot_parameters],
    )
    plt.close('all')
