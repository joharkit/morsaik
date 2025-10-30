import morsaik as kdi

def test_split_config():
    config_path = 'config/motif_trajectory_ensembles.yml'
    split_keys = ['mprc_id',]
    config = kdi.read.config(config_path)
    configs = kdi.utils.split_config(config, split_keys)
    # find out, how many config files and which parameters to split
    for key in config.keys():
        for ii in range(len(configs)):
            if key in split_keys:
                assert(config[key][ii]==configs[ii][key])
            else:
                assert(config[key]==configs[ii][key])

