import morsaik as kdi
import numpy as np

def test_make_hyphen_dct():
    motiflength = 4
    alphabet = ['a','b','c']
    actual_hyphen_dct = kdi.domains.make_hyphen_dct(alphabet,motiflength)
    desired_keys = ['length2strand_1','beginning_1','continuation_2','end_2']
    desired_wordlengths = [2,3,4,3]
    for ii in range(len(desired_keys)):
        desired_key = desired_keys[ii]
        desired_wordlength = desired_wordlengths[ii]
        assert(actual_hyphen_dct[desired_key].wordlength==desired_wordlength)
    for key in actual_hyphen_dct.keys():
        assert(key in desired_keys)
        
def test_alphabet():
    alphabet = ['a','b','c']
    motiflength = 5
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    assert(ms.alphabet==alphabet)


def test_alphabet():
    alphabet = ['a','b','c']
    motiflength = 5
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.alphabet==alphabet)

    alphabet = ['a','b','c']
    motiflength = 5
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.alphabet==alphabet)

def test_motiflength():
    alphabet = ['a','b','c']
    motiflength = 5
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.motiflength==motiflength)

    alphabet = ['a','b','c']
    motiflength = 5
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.motiflength==motiflength)

def test_number_of_letters():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.number_of_letters==number_of_letters)

    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.number_of_letters==number_of_letters)

def test_subspaces():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    for strandlength in range(2,motiflength-2):
        for hyphen_spot in range(1,strandlength):
            assert('length{ii}strand_{jj}'.format(ii=strandlength,jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in range(1,motiflength-motiflength//2):
        assert('beginning_{jj}'.format(jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in [motiflength-motiflength//2]:
        assert('continuation_{jj}'.format(jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in range(motiflength-motiflength//2,motiflength-1):
        assert('end_{jj}'.format(jj=hyphen_spot) in mhs.keys())

    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    for strandlength in range(2,motiflength-2):
        for hyphen_spot in range(1,strandlength):
            assert('length{ii}strand_{jj}'.format(ii=strandlength,jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in range(1,motiflength-motiflength//2):
        assert('beginning_{jj}'.format(jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in [motiflength-motiflength//2]:
        assert('continuation_{jj}'.format(jj=hyphen_spot) in mhs.keys())
    for hyphen_spot in range(motiflength-motiflength//2,motiflength-1):
        assert('end_{jj}'.format(jj=hyphen_spot) in mhs.keys())

def test_unit():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    mhs = kdi.domains.MotifHyphenSpace.make(alphabet, motiflength)
    assert(mhs.units/kdi.make_unit('bits', np.log2(len(alphabet)))==1.0)
