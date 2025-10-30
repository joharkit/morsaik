import morsaik as kdi
import nifty8 as ift

def test_trajectory_space():
    ms = kdi.domains.MotifSpace.make(['a','b'],4)
    ts = kdi.domains.TrajectorySpace(ms,ift.RGSpace(12))
    assert(ms.keys()==ts.keys())
