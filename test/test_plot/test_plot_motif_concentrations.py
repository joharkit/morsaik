import morsaik as kdi

def test_plot_motif_trajectories():
    motiflength = 4
    strand_trajectory_id = '9999_99_99__99_99_99'
    param_file_no = 0
    mte = kdi.get.strand_motifs_trajectory_ensemble(motiflength, strand_trajectory_id, param_file_no)
    c_ref = kdi.get.strand_reactor_parameters(strand_trajectory_id, param_file_no=param_file_no)
    kdi.plot.motif_trajectories(mte)
    plot_parameters = {
        'linestyle' : '-',
        'color' : 'b',
        'alpha' : 0.3,
        'label' : 'Strand',
        }
    kdi.plot.motif_trajectories(mte, plot_parameters=plot_parameters)
    kdi.plot.motif_trajectories([mte]*2)
    kdi.plot.motif_trajectories([mte]*2, plot_parameters=[plot_parameters]*2)
