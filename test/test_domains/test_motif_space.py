import morsaik as kdi
import numpy as np

def test__return_motif_categories():
    assert(('length{}strand','beginning','continuation','end')==kdi.domains._return_motif_categories())

def test__create_empty_motif_vector_dct():
    motiflength = 4
    alphabet = ['a','b','c']
    empty_motif_vector = kdi._create_empty_motif_vector_dct(motiflength,
            alphabet = alphabet)
    motif_categories = kdi.domains._return_motif_categories(motiflength=motiflength)
    for ii in range(len(motif_categories)):
        if ii<(len(motif_categories)-3):
            shape = (len(alphabet),)*(ii+1)
        elif ii==(len(motif_categories)-2) or ii==len(motif_categories):
            shape = (len(alphabet),)*(motiflength-1)
        else:
            shape = (len(alphabet),)*motiflength
        assert(np.all(empty_motif_vector[motif_categories[ii]]==np.zeros(shape)))

def test_alphabet():
    alphabet = ['a','b','c']
    motiflength = 5
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    assert(ms.alphabet==alphabet)

    alphabet = ['a','b','c']
    motiflength = 5
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=False)
    assert(ms.alphabet==alphabet)

def test_motiflength():
    alphabet = ['a','b','c']
    motiflength = 5
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    assert(ms.motiflength==motiflength)

    alphabet = ['a','b','c']
    motiflength = 5
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=False)
    assert(ms.motiflength==motiflength)

def test_number_of_letters():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    assert(ms.number_of_letters==number_of_letters)

    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=False)
    assert(ms.number_of_letters==number_of_letters)

def test_subspaces():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    for ii in range(1,motiflength-2):
        assert('length{ii}strand'.format(ii=ii) in ms.keys())
    assert('end' in ms.keys())
    assert('continuation' in ms.keys())
    assert('beginning' in ms.keys())

    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=False)
    assert('length{ii}strand'.format(ii=1) not in ms.keys())
    for ii in range(2,motiflength-2):
        assert('length{ii}strand'.format(ii=ii) in ms.keys())
    assert('end' in ms.keys())
    assert('continuation' in ms.keys())
    assert('beginning' in ms.keys())

def test_unit():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    ms = kdi.domains.MotifSpace.make(alphabet, motiflength,
            monomers_included=True)
    assert(ms.units/kdi.make_unit('bits', np.log2(len(alphabet)))==1.0)
