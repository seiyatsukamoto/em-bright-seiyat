import pytest

from ..gwlc import gwlc_functions

# m1, m2 values for Alsing, Farrow, Zhu initial NS/BH mass dists
# results produced by running function with the params below
als_result = [[1.53205977, 1.76327504, 1.68504319],
              [1.52401159, 1.40359535, 1.62760025]]

far_result = [[1.75185058, 2.44593256, 1.99912136],
              [1.3426239, 1.53290611, 1.65136978]]

zhu_result = [[8.24086231, 9.91239175, 6.84081889],
              [1.33484312, 1.3357914, 1.32818417]]


@pytest.mark.parametrize(
    'Type, result',
    [['BNS_alsing', als_result],
     ['BNS_farrow', far_result],
     ['NSBH_zhu', zhu_result],
     ]
)
def test_initial_mass_draws(Type, result):
    # five initial mass draws for unit test
    mass_draws_test = 3
    output = gwlc_functions.initial_mass_draws(Type, mass_draws_test)
    m1, m2 = output[0], output[1]
    # check component mass values
    assert (m1 == result[0]).all
    assert (m2 == result[1]).all
