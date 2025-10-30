import morsaik as kdi
import numpy as np
import nifty8 as ift

def test_time_space():
    times1 = np.arange(128)
    ts1 = kdi.domains.TimeSpace(times1,'s')
    rgs1 = ift.RGSpace(128,distances=1.)
    assert ts1.shape==rgs1.shape
    assert ts1.distances == rgs1.distances
    assert ts1.harmonic==rgs1.harmonic
    assert(kdi.domains.TimeSpace(times1,'s').units==kdi.make_unit('s'))
    times2 = np.arange(128)
    times2[-64:]*=10
    ts2 = kdi.domains.TimeSpace(times2,'s')
    ud2 = ift.UnstructuredDomain(128)
    assert ts2.units==kdi.make_unit('s')
    assert ts2.shape==ud2.shape
    assert(ts2.units==kdi.make_unit('s'))
    assert(not kdi.domains.are_compatible_timespaces(ts1,ts2))
    print(ts2.units)
    times3 = np.arange(128)
    times3[-64:]*=10
    ts3 = kdi.domains.TimeSpace(times3,'ms')
    print(ts2.units)
    print(ts3.units)
    assert(not kdi.domains.are_compatible_timespaces(ts1,ts3))
    print(ts2)
    print(ts3)
    print(ts2.units)
    print(ts3.units)
    assert(not kdi.domains.are_compatible_timespaces(ts2,ts3))
