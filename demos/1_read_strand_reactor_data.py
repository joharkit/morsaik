import morsaik as kdi
import numpy as np
from os import makedirs

if __name__ == '__main__':
    np.random.seed(42)

    # read configs
    config_folder_path = "./config"
    plotpath = './Plots/'
    makedirs(plotpath, exist_ok=True)
    strand_trajectory_ensembles_config = kdi.read.config(config_folder_path+"/strand_trajectory_ensembles.yml")

    strand_trj_ensemble_split_keys = ['param_file_no',]
    strand_trajectory_ensemble_configs = kdi.utils.split_config(
            strand_trajectory_ensembles_config,
            strand_trj_ensemble_split_keys
            )

    for run_no in range(len(strand_trajectory_ensemble_configs)):
        strand_trajectory_ensemble_config = strand_trajectory_ensemble_configs[run_no]

        # read test strand trajectories
        test_strand_trajectory_ensemble = kdi.get.strand_motifs_trajectory_ensemble(**strand_trajectory_ensemble_config)
        kdi.plot.motif_trajectories(
                [test_strand_trajectory_ensemble],
                **strand_trajectory_ensembles_config
                )
