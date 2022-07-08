import pytest
import numpy as np
from unittest.mock import patch
from gwemlightcurves.KNModels import KNTable
from ..lightcurves import calc_lightcurves
from ..lightcurves.mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu
from ..lightcurves.lightcurve_utils import load_EOS_posterior


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
    # three initial mass draws for unit test
    mass_draws_test = 3
    output = calc_lightcurves.initial_mass_draws(dist, mass_draws_test)
    m1, m2 = output[0], output[1]
    # check component mass values exist
    assert(len(m1) > 0)
    assert(len(m2) > 0)
    # check component mass values
    for m, r in zip(m1, result[0]):
        assert(np.abs(m - r) < 1e-6)
    for m, r in zip(m2, result[1]):
        assert(np.abs(m - r) < 1e-6)


dyn_result = [0.01858454429603665, 0.020577048962047758,
              0.024680568573388153, 0.024708485066836983,
              0.01865444152290588, 0.023558982026319016,
              0.0, 0.01312538012541829, 0.0, 0.0]

wind_result = [0.00015, 0.00015, 0.018125212769565864,
               0.019958112321618027, 0.00015,
               0.026042143314813155, 0.001797551492723432,
               0.00015, 0.0, 0.0]


@pytest.mark.parametrize(
    'm1, m2, thetas, wind_result, dyn_result',
    [[np.array([2.1]), np.array([1.5]), np.array([45]),
      wind_result, dyn_result]]
)
def test_run_EOS(m1, m2, thetas, wind_result, dyn_result):
    draws = load_EOS_posterior()
    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = calc_lightcurves.run_EOS(m1, m2, thetas,
                                       N_EOS=N_draws, EOS_draws=draws)
    wind_mej, dyn_mej = samples['wind_mej'], samples['dyn_mej']
    # check wind and dyn mej values exist
    assert(len(wind_mej) > 0)
    assert(len(dyn_mej) > 0)
    # check wind and dyn mej values
    for m, r in zip(wind_mej, wind_result):
        assert(np.abs(m - r) < 1e-6)
    for m, r in zip(dyn_mej, dyn_result):
        assert(np.abs(m - r) < 1e-6)

    # check that merger type matches NS/BH definition from EOS
    for sample in samples:
        if sample['merger_type'] == 1:
            assert (samples['m1'] <= samples['mbns']) & (samples['m2'] <= samples['mbns'])  # noqa:E501
        elif sample['merger_type'] == 2:
            assert (samples['m1'] > samples['mbns']) & (samples['m2'] <= samples['mbns'])  # noqa:E501
        elif sample['merger_type'] == 3:
            assert (samples['m1'] > samples['mbns']) & (samples['m2'] > samples['mbns'])  # noqa:E501


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([10.0]), np.array([1.5]), np.array([45])]]
)
def test_high_mass_ratio(m1, m2, thetas):
    draws = load_EOS_posterior()
    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = calc_lightcurves.run_EOS(m1, m2, thetas,
                                       N_EOS=N_draws, EOS_draws=draws)
    mej = samples['mej']
    # check high mass ratio has low mej
    assert (np.mean(mej) < 1e-3)


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([1.4]), np.array([1.4]), np.array([45])]]
)
def test_low_mass_wind_check(m1, m2, thetas):
    draws = load_EOS_posterior()
    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = calc_lightcurves.run_EOS(m1, m2, thetas,
                                       N_EOS=N_draws, EOS_draws=draws)
    wind_mej = samples['wind_mej']
    # low mass BNS should have significant wind mej
    assert (np.mean(wind_mej) > 1e-3)


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([2.4]), np.array([2.4]), np.array([45])]]
)
def test_high_mass_wind_check(m1, m2, thetas):
    draws = load_EOS_posterior()
    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = calc_lightcurves.run_EOS(m1, m2, thetas,
                                       N_EOS=N_draws, EOS_draws=draws)
    wind_mej = samples['wind_mej']
    # high mass BNS should have small wind mej
    assert (np.mean(wind_mej) < 1e-3)


@pytest.mark.parametrize(
    'samples',
    [KNTable(([.1], [35], [0]), names=('mej', 'theta', 'sample_id'))]
)
def test_ejecta_to_lc(samples):
    mags = [np.ones((9, 500))]
    t = [np.ones(500)]
    mock_mags = KNTable((t, mags), names=('t', 'mag'))
    with patch('gwemlightcurves.KNModels.KNTable.model') as mock_KNTable:
        mock_KNTable.return_value = mock_mags
        lightcurve_data = calc_lightcurves.ejecta_to_lc(samples)
        # check the mock object was called once, and has the right value
        mock_KNTable.assert_called_once()
        assert mock_KNTable.return_value is mock_mags
    # check if all 9 bands are present
    assert lightcurve_data['mag'].shape == (1, 9, 500)


# m1s = None, m2s = None, thetas = None, mass_dist = None, mass_draws = None
@pytest.mark.parametrize(
    'm1, m2, theta',
    [[np.array([1.5, 4.0]), np.array([1.5, 1.2]), np.array([45, 45])]]
)
def test_lightcurve_predictions(m1, m2, theta):
    lightcurve_data1, has_Remnant1 = calc_lightcurves.lightcurve_predictions(mass_dist=BNS_alsing, N_EOS=4, mass_draws=1)  # noqa:E501
    lightcurve_data2, has_Remnant2 = calc_lightcurves.lightcurve_predictions(m1s=m1, m2s=m2, thetas=theta, N_EOS=2)  # noqa:E501
    assert has_Remnant1 == 1.0
    assert has_Remnant2 == 0.5
