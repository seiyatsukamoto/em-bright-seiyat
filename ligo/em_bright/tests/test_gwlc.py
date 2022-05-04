import pytest
import numpy as np
from pathlib import Path
from astropy.table import Table
from unittest.mock import patch
from gwemlightcurves.KNModels import KNTable
from ..gwlc import lightcurves
from ..gwlc.mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu

# m1, m2 values for Alsing, Farrow, Zhu initial NS/BH mass dists
# results produced by running function with the params below
als_result = [[1.53205977, 1.76327504, 1.68504319],
              [1.52401159, 1.40359535, 1.62760025]]

far_result = [[1.75185058, 2.44593256, 1.99912136],
              [1.3426239, 1.53290611, 1.65136978]]

zhu_result = [[8.24086231, 9.91239175, 6.84081889],
              [1.33484312, 1.3357914, 1.32818417]]


@pytest.mark.parametrize(
    'dist, result',
    [[BNS_alsing, als_result],
     [BNS_farrow, far_result],
     [NSBH_zhu, zhu_result],
     ]
)
def test_initial_mass_draws(dist, result):
    # five initial mass draws for unit test
    mass_draws_test = 3
    output = lightcurves.initial_mass_draws(dist, mass_draws_test)
    m1, m2 = output[0], output[1]
    # check component mass values
    assert (m1 == result[0]).all
    assert (m2 == result[1]).all


result_gp10 = np.array([[0.022264876109183446, 0.024506102696707492],
                        [0.0018540989881778943, 0.0006251326630963076]])


@pytest.mark.parametrize(
    'EOS, m1, m2, thetas, result',
    [['gp', np.array([1.5]), np.array([1.5]), np.ones(10)*45, result_gp10]]
)
def test_run_EOS(EOS, m1, m2, thetas, result):
    post_path = 'data/eos_post_PSRs+GW170817+J0030.csv'
    with open(Path(__file__).parents[0] / post_path, 'rb') as f:
        post = np.genfromtxt(f, names=True, delimiter=",")
    # Note weights removed, files will be used from Zenodo outside of tests
    idxs = ['137', '138', '421', '422', '423',
            '424', '425', '426', '427', '428']

    draws = {}
    for idx in idxs:
        EOS_draw_path = f'data/MACROdraw-1151{idx}-0.csv'
        with open(Path(__file__).parents[0] / EOS_draw_path, 'rb') as f:
            draws[f'{idx}'] = np.genfromtxt(f, names=True, delimiter=",")

    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = lightcurves.run_EOS(EOS, m1, m2, thetas, N_EOS=N_draws, EOS_posterior=post, EOS_draws=draws, EOS_idx=idxs)
    wind_mej, dyn_mej = samples['wind_mej'], samples['dyn_mej']
    # check wind and dyn mej values
    assert (list(wind_mej[3:5]) == result[:, 1]).all
    assert (list(dyn_mej[3:5]) == result[:, 0]).all

@pytest.mark.parametrize(
    'samples, result',
    [[Table(([.1], [35], [0]), names=('mej', 'theta', 'sample_id')), [-16.640245217501654, -7.776421192396103, -10.355205145387737]]]
)
def test_ejecta_to_lc(samples, result):
    with patch('gwemlightcurves.KNModels.KNTable.model') as mock_KNTable:
        mags = [np.ones((9,500))]
        t = [np.ones(500)]
        #mej theta sample_id phi  tini tmax  dt vmin  th  ph  kappa      eps      alp eth flgbct beta kappa_r slope_r theta_r  Ye n_coeff  gptype mej10 t [500] lbol [500] mag [9,500]
        mock_KNTable.return_value = KNTable((t,mags), names=('t', 'mag'))
        lightcurve_data = lightcurves.ejecta_to_lc(samples)
        # check if all 9 bands are present
        assert lightcurve_data['mag'].shape == (1,9,500)
