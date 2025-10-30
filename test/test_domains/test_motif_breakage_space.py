import morsaik as kdi

def test_MotifBreakageSpace():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifBreakageSpace.make(alphabet, motiflength)
