import morsaik as kdi

def test_alphabet():
    alphabet = ['a','b','c']
    wordlength = 5
    hamming_space = kdi.domains.HammingSpace(alphabet, wordlength)
    assert(hamming_space.alphabet==alphabet)

def test_wordlength():
    alphabet = ['a','b','c']
    wordlength = 5
    hamming_space = kdi.domains.HammingSpace(alphabet, wordlength)
    assert(hamming_space.wordlength==wordlength)

def test_dvol():
    alphabet = ['a','b','c']
    wordlength = 5
    hamming_space = kdi.domains.HammingSpace(alphabet, wordlength)
    assert(hamming_space._dvol==1.)
