import morsaik as kdi
import jax.numpy as jnp
from itertools import product as iterprod
import pytest

@pytest.mark.skip
def test_motif_production_rate_constants_extension_matrix():
    number_of_letters = 2
    motiflength = hybridization_length_max = 4
    motif_production_rate_constants_extension_matrix = kdi.infer.motif_production_rate_constants_extension_matrix(
            number_of_letters, motiflength, hybridization_length_max
            )
    hybridization_configuration_categories, hybridization_configuration_indices = kdi.infer._hybridization_site_categories(number_of_letters, hybridization_length_max)
    motif_indices = jnp.concatenate([jnp.zeros(1),jnp.cumsum(number_of_letters**jnp.arange(1,motiflength+1)),jnp.array([jnp.sum(number_of_letters**jnp.arange(1,motiflength+1))+number_of_letters**(motiflength-1)])],dtype=int)
    template_indices = (motif_indices - motif_indices[1]).at[0].set(0)
    mprc = jnp.zeros(hybridization_configuration_indices[-1])
    for ii in hybridization_configuration_indices:
        mprc = mprc.at[ii].add(ii+1)
    emprc = motif_production_rate_constants_extension_matrix @ mprc
    dzero = emprc
    for ii in range(0, len(hybridization_configuration_categories)):
        a = hybridization_configuration_categories[ii]
        left_shift = a[-1]
        left_ligant_length = a[1]
        right_ligant_length = a[2]
        template_length = a[3]
        right_shift = left_shift + right_ligant_length + left_ligant_length - template_length
        if left_shift==-1:
            left_ligant_lengths = jnp.arange(a[1],motiflength)
            # blunt end gets extended iff hybridization_length=hybridization_length_max and the ligation spot is at the center or further apart.
            # for the other part of the complex, the same rules apply as for every hybridization_length: for a dangling end, the dangling strand/motif can be extended
            # for a blunt ent that is close to the ligation spot, the complex is supposed to end there
            # This way, we only extend blunt ends on both sides, if the ligation_spot is at the center
            # If hybridization_length_max is uneven, the right central ligation spot is treated as the center,
            # so the center is at hybridization_length_max-hybridization_length_max//2 from the left side and hybridization_lengh_max//2 from the right side apart 
        elif (a[0]==hybridization_length_max) and (left_shift==0) and (a[1]>=hybridization_length_max-hybridization_length_max//2):
            # continue left ligant
            left_ligant_lengths = jnp.arange(a[1],motiflength)
        else:
            left_ligant_lengths = [a[1]]
        if right_shift==1:
            right_ligant_lengths = jnp.arange(a[2],motiflength)
        elif (a[0]==hybridization_length_max) and (right_shift==0) and (a[2]>=hybridization_length_max//2):
            right_ligant_lengths = jnp.arange(a[2],motiflength)
        else:
            right_ligant_lengths = [a[2]]
        template_might_continue_forwards = (left_shift==1) or ((a[0]==hybridization_length_max) and (left_shift==0))
        template_might_continue_backwards = (right_shift==-1) or ((a[0]==hybridization_length_max) and (right_shift==0))
        template_might_continue_both_ways = (template_might_continue_forwards and template_might_continue_backwards)
        if template_might_continue_both_ways:
            template_overhangs = jnp.arange(motiflength-a[3])
        elif template_might_continue_forwards: #(but not backwards)
            template_overhangs = jnp.arange(motiflength-a[3])
        else:
            template_overhangs = jnp.arange(1)
        for left_ligant_length, right_ligant_length, template_overhang in iterprod(left_ligant_lengths, right_ligant_lengths, template_overhangs):
            if template_overhang+a[3] == motiflength:
                template_lengths = jnp.array([motiflength])
            elif template_might_continue_backwards or template_might_continue_forwards:
                template_lengths = jnp.arange(a[3]+template_overhang, motiflength)
            else:
                template_lengths = jnp.array([a[3]+template_overhang])
            for template_length in template_lengths:
                if (template_length == (motiflength-1)) and template_might_continue_backwards and not template_might_continue_forwards:
                    template_index_start = template_indices[-1-1]
                else:
                    template_index_start = template_indices[template_length-1]
                dzero = dzero.at[
                        motif_indices[left_ligant_length-1],
                        motif_indices[right_ligant_length-1],
                        template_indices[template_length-1]
                        ].add(-ii-1)
                print(f'{int(motif_indices[left_ligant_length-1]), int(motif_indices[right_ligant_length-1]), int(template_indices[template_length-1])=}')
    # 01) (2, 1, 1, 2, 0)
    # 02) (3, 1, 1, 3, 0),
    # 03) (3, 1, 1, 3, 1),
    # 04) (3, 1, 2, 2, 0),
    # 05) (3, 1, 2, 3, 0),
    # 06) (3, 2, 1, 2, -1),
    # 07) (3, 2, 1, 3, 0),
    # 08) (4, 1, 1, 4, 1),
    # 09) (4, 1, 2, 3, 1),
    # 10) (4, 1, 2, 4, 0),
    # 11) (4, 1, 2, 4, 1),
    # 12) (4, 1, 3, 3, 0),
    # 13) (4, 1, 3, 4, 0),
    # 14) (4, 2, 1, 3, -1),
    # 15) (4, 2, 1, 4, 0),
    # 16) (4, 2, 1, 4, 1),
    # 17) (4, 2, 2, 2, -1),
    # 18) (4, 2, 2, 3, -1),
    # 19) (4, 2, 2, 3, 0),
    # 20) (4, 2, 2, 4, 0),
    # 21) (4, 3, 1, 3, -1),
    # 22) (4, 3, 1, 4, 0)
    print(f'{jnp.array(list(jnp.nonzero(dzero))).T=}')
    print(f'{jnp.array(list(jnp.nonzero(emprc))).T=}')
    assert jnp.allclose(dzero, 0.)
