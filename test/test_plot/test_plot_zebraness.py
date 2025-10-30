import morsaik as kdi
import matplotlib.pyplot as plt

def test_plot_zebraness():
    motiflength = 4
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0

    motif_trajectory_ensemble = kdi.get.strand_motifs_trajectory_ensemble(
        motiflength,
        strand_trajectory_id,
        param_file_no=0
    )

    plt.close('all')
    kdi.plot.zebraness(motif_trajectory_ensemble)
    plt.close('all')


def test_plot_system_level_motif_zebraness():
    motiflength = 4
    strand_trajectory_id='9999_99_99__99_99_99'
    param_file_no=0

    motif_trajectory_ensemble = kdi.get.strand_motifs_trajectory_ensemble(
        motiflength,
        strand_trajectory_id,
        param_file_no=0
    )

    plt.close('all')
    kdi.plot.system_level_motif_zebraness(motif_trajectory_ensemble)
    plt.close('all')
