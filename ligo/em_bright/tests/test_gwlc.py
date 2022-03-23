from pathlib import Path
import pytest
import numpy as np
from ..gwlc import gwlc_functions
# from mass_grid_gp10_test2 import run_EOS

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

result_gp10 = np.array([[0.022264876109183446, 0.024506102696707492],
                        [0.0018540989881778943, 0.0006251326630963076]])

# once full EOS implementaion is finished update np.ones(10)
@pytest.mark.parametrize(
    'EOS, m1, m2, thetas, EOS, result',
    [['gp', np.array([1.5]), np.array([1.5]), np.ones(10)*45, result_gp10]]
)


def test_run_EOS(EOS, m1, m2, thetas, result):
    post_path = 'data/eos_post_PSRs+GW170817+J0030.csv'
    with open(Path(__file__).parents[0] / post_path, 'rb') as f:
        post = np.genfromtxt(f, names=True, delimiter=",")
    # Note: make sure original weighting gets re-implented for real case
    idxs = ['137', '138', '421', '422', '423',
            '424', '425', '426', '427', '428']

    draws = {}
    for idx in idxs:
        print('EOS file:', idx)
        EOS_draw_path = f'data/MACROdraw-1151{idx}-0.csv'
        with open(Path(__file__).parents[0] / EOS_draw_path, 'rb') as f:
            draws[f'{idx}'] = np.genfromtxt(f, names=True, delimiter=",")

    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = run_EOS(EOS, m1, m2, thetas, N_EOS = N_draws, eospostdat = post, EOS_draws = draws, EOS_idx = idxs)
    wind_mej, dyn_mej = samples['wind_mej'], samples['dyn_mej']
    # check wind and dyn mej values
    assert (list(wind_mej[3:5]) == result[:, 1]).all
    assert (list(dyn_mej[3:5]) == result[:, 0]).all

# test_run_EOS('gp', np.array([1.5]), np.array([1.5]), np.ones(10)*45, result_gp10)
