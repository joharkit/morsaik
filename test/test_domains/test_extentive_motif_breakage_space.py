import morsaik as kdi

def test_subdomains():
    alphabet = ['a','b','c']
    motiflength = 5
    number_of_letters = len(alphabet)
    embs = kdi.domains.makeExtentiveMotifBreakageSpace(alphabet, motiflength)

    ending_categories = ['length{}strand'.format(ii) for ii in range(1,motiflength-1)] + ['end']
    beginning_categories = ['length{}strand'.format(ii) for ii in range(1,motiflength-1)] + ['beginning']
    desired_keys = []
    for ending_category in ending_categories:
        for beginning_category in beginning_categories:
            desired_keys += [ending_category + '-' + beginning_category]
    assert(set(desired_keys) == set(embs.keys()))
