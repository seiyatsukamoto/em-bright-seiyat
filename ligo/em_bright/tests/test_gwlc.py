import pytest
import numpy as np
import h5py
from pathlib import Path
from astropy.table import Table
from unittest.mock import patch
from gwemlightcurves.KNModels import KNTable
from ..lightcurves import calc_lightcurves
from ..lightcurves.mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu

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

# mej results generated from samples of 10 EOS's for unit test
wind_result = [0.00015, 0.00015, 0.00015, 0.00015, 0.01704984942589736,
               0.00015, 0.00015, 0.00015, 0.0007670940412947962, 0.00015]

dyn_result = [0.00359207322927068, 0.00359207322927068, 0.00474938064347107, 0.004139254128378871, 0.002229370192200336,
              0.0037482586921199225, 0.0037482586921199225, 0.004139254128378871, 0.002969051216059871, 0.00474938064347107]

@pytest.mark.parametrize(
    'm1, m2, thetas, wind_result, dyn_result',
    [[np.array([1.5]), np.array([1.5]), np.ones(10)*45, wind_result, dyn_result]]
)
def test_run_EOS(m1, m2, thetas, wind_result, dyn_result):
    # draw from subset of EOS's for unit test
    EOS_draw_path = 'data/10_EOS_unit_test.h5'
    with open(Path(__file__).parents[0] / EOS_draw_path, 'rb') as f:
        dset = h5py.File(f, 'r')
        draws = np.array(dset['EOS'])
    # number of EOS draws, in this case the number of EOS files
    N_draws = 10
    samples = calc_lightcurves.run_EOS(m1, m2, thetas, N_EOS=N_draws, EOS_draws=draws)
    wind_mej, dyn_mej = samples['wind_mej'], samples['dyn_mej']
    # check wind and dyn mej values exist
    assert(len(wind_mej) > 0)
    assert(len(dyn_mej) > 0)
    # check wind and dyn mej values
    for m, r in zip(wind_mej, wind_result):
        assert(np.abs(m - r) < 1e-6)
    for m, r in zip(dyn_mej, dyn_result):
        assert(np.abs(m - r) < 1e-6)

@pytest.mark.parametrize(
    'samples',
    [Table(([.1], [35], [0]), names=('mej', 'theta', 'sample_id'))]
)
def test_ejecta_to_lc(samples):
    with patch('gwemlightcurves.KNModels.KNTable.model') as mock_KNTable:
        mags = [np.ones((9,500))]
        t = [np.ones(500)]
        #mej theta sample_id phi  tini tmax  dt vmin  th  ph  kappa      eps      alp eth flgbct beta kappa_r slope_r theta_r  Ye n_coeff  gptype mej10 t [500] lbol [500] mag [9,500]
        mock_KNTable.return_value = KNTable((t,mags), names=('t', 'mag'))
        lightcurve_data = calc_lightcurves.ejecta_to_lc(samples)
        # check if all 9 bands are present
        assert lightcurve_data['mag'].shape == (1,9,500)
