import morsaik as kdi
import pytest
from numpy.testing import assert_equal
import itertools

def test_alphabet():
    alphabet = ['a','b','c']
    motiflength = 5
    maximum_ligation_window_length = motiflength
    mps = kdi.domains.MotifProductionSpace.make(alphabet, motiflength,
            maximum_ligation_window_length)
    assert(mps.alphabet==alphabet)

def test_motiflength():
    alphabet = ['a','b','c']
    motiflength = 5
    maximum_ligation_window_length = motiflength
    mps = kdi.domains.MotifProductionSpace.make(alphabet, motiflength,
            maximum_ligation_window_length)
    assert(mps.motiflength==motiflength)

def test_number_of_letters():
    alphabet = ['a','b','c']
    motiflength = 5
    maximum_ligation_window_length = motiflength
    number_of_letters = len(alphabet)
    mps = kdi.domains.MotifProductionSpace.make(alphabet, motiflength,
            maximum_ligation_window_length)
    assert(mps.number_of_letters==number_of_letters)

def test_make_motif_production_dct():
    alphabet = ['a','b']
    motiflength = 4
    maximum_ligation_window_length = motiflength
    motif_categories = kdi.domains._return_motif_categories(motiflength)
    desired = []
    for product_category in motif_categories[1:]:
        for template_category in motif_categories[1:]:
            left_reactant_length = 1
            ligation_position = 1
            ligation_window_length = 4
            desired = desired + [kdi.domains._production_channel_id(product_category,template_category,ligation_window_length,ligation_position),]
    assert_equal(set(
            kdi.domains.make_motif_production_dct(alphabet,motiflength,
                maximum_ligation_window_length).keys()),
            set(desired))

    alphabet = ['a','b']
    motiflength = 6
    maximum_ligation_window_length = motiflength
    motif_categories = kdi.domains._return_motif_categories(motiflength)
    desired = []
    for ligation_window_length in range(4,motiflength+1):
        for ligation_spot in range(1,ligation_window_length-2):
            current_motif_categories = [mc.format(ligation_window_length-2) for mc in kdi.domains._motif_categories()]
            if ligation_window_length<maximum_ligation_window_length:
                skip_list = [(current_motif_categories[-2],current_motif_categories[-3])]
                skip_list = skip_list + [(current_motif_categories[-1],current_motif_categories[-3])]
                skip_list = skip_list + [(current_motif_categories[-3],current_motif_categories[-2])]
                skip_list = skip_list + [(current_motif_categories[-2],current_motif_categories[-2])]
                skip_list = skip_list + [(current_motif_categories[-1],current_motif_categories[-2])]
                skip_list = skip_list + [(current_motif_categories[-3],current_motif_categories[-1])]
                skip_list = skip_list + [(current_motif_categories[-2],current_motif_categories[-1])]
            elif (ligation_spot < (ligation_window_length-ligation_window_length//2-1)):
                #no beginnning product or ending template in skip_list
                skip_list = [(pc, tc) for pc in current_motif_categories[-2:] for tc in current_motif_categories[-3:-1]]
            elif (ligation_spot > (ligation_window_length-ligation_window_length//2-1)):
                #no ending product or beginnning template in skip_list
                skip_list = [(pc, tc) for pc in current_motif_categories[-3:-1] for tc in current_motif_categories[-2:]]
            else:
                skip_list = []
            for product_category, template_category in itertools.product(current_motif_categories, repeat = 2):
                if (product_category, template_category) in skip_list:
                    continue
                desired = desired + [kdi.domains._production_channel_id(product_category,template_category,ligation_window_length,ligation_spot),]
    print('keys that are not in desired')
    for key in kdi.domains.make_motif_production_dct(alphabet,motiflength, maximum_ligation_window_length).keys():
        if key not in desired:
            print(key)
    print('keys that are not in actual')
    for key in desired:
        if key not in kdi.domains.make_motif_production_dct(alphabet,motiflength, maximum_ligation_window_length).keys():
            print(key)
    assert_equal(set(
            kdi.domains.make_motif_production_dct(alphabet,motiflength,
                maximum_ligation_window_length).keys()),
            set(desired))

def test_subspaces():
    alphabet = ['a','b']
    motiflengths = [4,5,6,9,10]

    for motiflength in motiflengths:
        maximum_ligation_window_length = motiflength
        motif_categories = kdi.domains._return_motif_categories(motiflength)
        mps = kdi.domains.MotifProductionSpace.make(alphabet,motiflength,maximum_ligation_window_length)
        desired = []
        for ligation_window_length in range(4,motiflength+1):
            for ligation_spot in range(1,ligation_window_length-2):
                current_motif_categories = [mc.format(ligation_window_length-2) for mc in kdi.domains._motif_categories()]
                if ligation_window_length<maximum_ligation_window_length:
                    skip_list = [(current_motif_categories[-2],current_motif_categories[-3])]
                    skip_list = skip_list + [(current_motif_categories[-1],current_motif_categories[-3])]
                    skip_list = skip_list + [(current_motif_categories[-3],current_motif_categories[-2])]
                    skip_list = skip_list + [(current_motif_categories[-2],current_motif_categories[-2])]
                    skip_list = skip_list + [(current_motif_categories[-1],current_motif_categories[-2])]
                    skip_list = skip_list + [(current_motif_categories[-3],current_motif_categories[-1])]
                    skip_list = skip_list + [(current_motif_categories[-2],current_motif_categories[-1])]
                elif (ligation_spot < (ligation_window_length-ligation_window_length//2-1)):
                    #no beginnning product or ending template in skip_list
                    skip_list = [(pc, tc) for pc in current_motif_categories[-2:] for tc in current_motif_categories[-3:-1]]
                elif (ligation_spot > (ligation_window_length-ligation_window_length//2-1)):
                    #no ending product or beginnning template in skip_list
                    skip_list = [(pc, tc) for pc in current_motif_categories[-3:-1] for tc in current_motif_categories[-2:]]
                else:
                    skip_list = []
                for product_category, template_category in itertools.product(current_motif_categories, repeat = 2):
                    if (product_category, template_category) in skip_list:
                        continue
                    desired = desired + [kdi.domains._production_channel_id(product_category,template_category,ligation_window_length,ligation_spot),]
        # assert desired and actual are equal
        assert(set(desired)==set(mps.keys()))
