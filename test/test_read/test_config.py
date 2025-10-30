import morsaik as kdi

def test_read_config():
    config_path = 'config/mprc.yml'
    assert(isinstance(kdi.read.config(config_path),dict))
