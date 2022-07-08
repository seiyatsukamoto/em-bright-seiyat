from pathlib import Path
import pytest
import numpy as np
from astropy.table import Table
from unittest.mock import patch
from gwemlightcurves.KNModels import KNTable
from ..lightcurves import calc_lightcurves
from ..lightcurves.mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu
from ..lightcurves.mass_distributions import BNS_uniform, BNS_LRR
from ..lightcurves.mass_distributions import NSBH_uniform, NSBH_LRR
from ..lightcurves.lightcurve_utils import load_eos_posterior

from ..lightcurves.lightcurve_utils import load_EOS_posterior

# m1, m2 values for Alsing, Farrow, Zhu initial NS/BH mass dists
# results produced by running function with the params below
als_result = [[1.53205977, 1.76327504, 1.68504319],
              [1.52401159, 1.40359535, 1.62760025]]

far_result = [[1.75185058, 2.44593256, 1.99912136],
              [1.3426239, 1.53290611, 1.65136978]]

zhu_result = [[8.24086231, 9.91239175, 6.84081889],
              [1.33484312, 1.3357914, 1.32818417]]

BNS_uni_result = [[2.4268996, 2.70593716, 2.18786574],
                  [1.34370059, 1.30106652, 1.98714215]]

BNS_LRR_result = [[1.45691842, 1.46629522, 1.30002483],
                  [1.30820654, 1.1892853, 1.27899721]]

NSBH_uni_result = [[6.08817749, 6.06047861, 6.08466998],
                   [2.70437173, 2.25909451, 1.71116343]]

NSBH_LRR_result = [[1.55573485, 2.50949848, 1.43821419],
                   [1.33426283, 1.44072616, 1.04894091]]


@pytest.mark.parametrize(
    'dist, result',
    [[BNS_alsing, als_result],
     [BNS_farrow, far_result],
     [NSBH_zhu, zhu_result],
     [BNS_uniform, BNS_uni_result],
     [BNS_LRR, BNS_LRR_result],
     [NSBH_uniform, NSBH_uni_result],
     [NSBH_LRR, NSBH_LRR_result],
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


wind_result = [0.0256483901004048, 0.00015, 0.00015,
               0.0, 0.0, 0.0, 0.0, 0.00015,
               0.024255867289002035, 0.00015]

dyn_result = [0.026983497953596173, 0.02016999499804752,
              0.01770449656270658, 0.0, 0.0, 0.0, 0.0,
              0.023883542614886615, 0.02623057446187879,
              0.013457388566138451]


@pytest.mark.parametrize(
    'm1, m2, thetas, wind_result, dyn_result',
    [[np.array([2.1]), np.array([1.5]), np.array([45]),
      wind_result, dyn_result]]
)
def test_run_eos(m1, m2, thetas, wind_result, dyn_result):
    draws = load_eos_posterior()
    # number of eos draws, in this case the number of eos files
    N_draws = 10
    samples, _ = calc_lightcurves.run_eos(m1, m2, thetas, N_eos=N_draws,
                                          eos_draws=draws)
    wind_mej, dyn_mej = samples['wind_mej'], samples['dyn_mej']
    # check wind and dyn mej values exist
    assert(len(wind_mej) > 0)
    assert(len(dyn_mej) > 0)
    # check wind and dyn mej values
    for m, r in zip(wind_mej, wind_result):
        assert(np.abs(m - r) < 1e-6)
    for m, r in zip(dyn_mej, dyn_result):
        assert(np.abs(m - r) < 1e-6)

    # check that merger type matches NS/BH definition from eos
    for sample in samples:
        if sample['merger_type'] == 'BNS':
            assert (sample['m1'] <= sample['mbns']) & (sample['m2'] <= sample['mbns'])  # noqa:E501
        elif sample['merger_type'] == 'NSBH':
            assert (sample['m1'] > sample['mbns']) & (sample['m2'] <= sample['mbns'])  # noqa:E501
        elif sample['merger_type'] == 'BBH':
            assert (sample['m1'] > sample['mbns']) & (sample['m2'] > sample['mbns'])  # noqa:E501


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([10.0]), np.array([1.5]), np.array([45])]]
)
def test_high_mass_ratio(m1, m2, thetas):
    # a high mass ratio merger should produce very little mass ejecta
    draws = load_eos_posterior()
    # number of eos draws, in this case the number of eos files
    N_draws = 10
    samples, _ = calc_lightcurves.run_eos(m1, m2, thetas, N_eos=N_draws,
                                          eos_draws=draws)
    mej = samples['mej']
    # check high mass ratio has low mej on average
    assert (np.mean(mej) < 1e-3)


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([1.4]), np.array([1.4]), np.array([45])]]
)
def test_low_mass_wind_check(m1, m2, thetas):
    # a low mass BNS merger should produce significant wind ejecta
    draws = load_eos_posterior()
    # number of eos draws, in this case the number of eos files
    N_draws = 10
    samples, _ = calc_lightcurves.run_eos(m1, m2, thetas, N_eos=N_draws,
                                          eos_draws=draws)
    wind_mej = samples['wind_mej']
    # check for significant wind mej on average
    assert (np.mean(wind_mej) > 1e-3)


@pytest.mark.parametrize(
    'm1, m2, thetas',
    [[np.array([2.4]), np.array([2.4]), np.array([45])]]
)
def test_high_mass_wind_check(m1, m2, thetas):
    # a high mass BNS merger should produce little wind ejecta
    draws = load_eos_posterior()
    # number of eos draws, in this case the number of eos files
    N_draws = 10
    samples, _ = calc_lightcurves.run_eos(m1, m2, thetas, N_eos=N_draws,
                                          eos_draws=draws)
    wind_mej = samples['wind_mej']
    # check for minimal wind ejecta
    assert (np.mean(wind_mej) < 1e-3)


@pytest.mark.parametrize(
    'samples',
    [KNTable(([.1], [35], [0]), names=('mej', 'theta', 'sample_id'))]
)
def test_ejecta_to_lightcurve(samples):
    mags = [np.ones((9, 500))]
    t = [np.ones(500)]
    mock_mags = KNTable((t, mags), names=('t', 'mag'))
    with patch('gwemlightcurves.KNModels.KNTable.model') as mock_KNTable:
        mock_KNTable.return_value = mock_mags
        lightcurve_data, _ = calc_lightcurves.ejecta_to_lightcurve(samples)
        # check the mock object was called once, and has the right value
        mock_KNTable.assert_called_once()
        assert mock_KNTable.return_value is mock_mags
    # check if all 9 bands are present
    assert lightcurve_data['mag'].shape == (1, 9, 500)


u_band_percentiles = [-13.19872909, -13.09596377, -2.63326588]
r_band_percentiles = [-14.64907056, -14.56544989,  -4.61280546]
mej_percentiles = [1.0e-11, 3.61024744e-03, 4.09925496e-03]


@pytest.mark.parametrize(
    'm1, m2, theta, yields_ejecta, percentile_results',
    [[np.array([1.5, 4.0]), np.array([1.5, 1.2]),
      np.array([45, 45]), [1.0, 0.7], {'mej': mej_percentiles,
      'u': u_band_percentiles, 'r': r_band_percentiles}]]
)
def test_lightcurve_predictions(m1, m2, theta, yields_ejecta, percentile_results):  # noqa:E501
    # check that yields_ejecta is working both for a mass
    # distribution and when passing specific masses
    # also check percentiles are working for mag and mej
    # BNS alsing will usually produce significant ejecta
    lightcurve_data1, yields_ejecta1, _, _ = calc_lightcurves.lightcurve_predictions(mass_dist=BNS_alsing, N_eos=2, mass_draws=2)  # noqa:E501
    # NSBH mergers will more frequently produce minimal mass ejecta
    lightcurve_data2, yields_ejecta2, _, _ = calc_lightcurves.lightcurve_predictions(m1s=m1, m2s=m2, thetas=theta, N_eos=5)  # noqa:E501
    assert yields_ejecta1 == yields_ejecta[0]
    assert yields_ejecta2 == yields_ejecta[1]
    percentiles = calc_lightcurves.find_percentiles(lightcurve_data2)
    percentile_types = ['mej', 'u', 'r']
    for percentile_type in percentile_types:
        for percentile, result in zip(percentiles[percentile_type], percentile_results[percentile_type]):  # noqa:E501
            assert abs(percentile-result) < 1e-6

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
>>>>>>> 543f0ca (Additional tests and changes)

result_gp10 = np.array([[0.022264876109183446, 0.024506102696707492],
                        [0.0018540989881778943, 0.0006251326630963076]])

# once full EOS implementaion is finished update np.ones(10)
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

    draws = {}
    for idx in idxs:
        print('EOS file:', idx)
        EOS_draw_path = f'data/MACROdraw-1151{idx}-0.csv'
        with open(Path(__file__).parents[0] / EOS_draw_path, 'rb') as f:
            draws[f'{idx}'] = np.genfromtxt(f, names=True, delimiter=",")

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

def test_lightcurve_predictions(m1, m2, theta):
    # check that has_Remnant is working both for a mass
    # distribution and when passing specific masses
    # BNS alsing will usually produce significant ejecta
    lightcurve_data1, has_Remnant1, _, _ = calc_lightcurves.lightcurve_predictions(mass_dist=BNS_alsing, N_eos=2, mass_draws=2)  # noqa:E501
    # NSBH mergers will frequently produce minimal mass ejecta
    lightcurve_data2, has_Remnant2, _, _ = calc_lightcurves.lightcurve_predictions(m1s=m1, m2s=m2, thetas=theta, N_eos=2)  # noqa:E501
    assert has_Remnant1 == 1.0
    assert has_Remnant2 == 0.5
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
