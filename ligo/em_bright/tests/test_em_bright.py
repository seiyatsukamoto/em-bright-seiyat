from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import h5py
import pandas as pd

import pytest
from unittest.mock import Mock
from scipy.interpolate import interp1d

from .. import em_bright, categorize, utils, EOS_MAX_MASS


def test_version():
    from .. import __version__
    assert __version__ == '1.1.0.dev1'


@pytest.mark.parametrize(
    'posteriors, dtype, result, result_eos',
    [[[(1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 100.0),
       (2.0, 0.5, 0.99, 0.99, 0.0, 0.0, 150.0)],
      [('chirp_mass', '<f8'), ('mass_ratio', '<f8'), ('a_1', '<f8'),
       ('a_2', '<f8'), ('tilt_1', '<f8'), ('tilt_2', '<f8'),
       ('luminosity_distance', '<f8')],
      (1.0, 1.0, 0.5), (1.0, 0.5, 0.5)],
     [[(1.2, 1.0, 0.0, 0.0, 100.0),
       (2.0, 0.5, 0.99, 0.99, 150.0)],
      [('chirp_mass', '<f8'), ('mass_ratio', '<f8'), ('a_1', '<f8'),
       ('a_2', '<f8'), ('luminosity_distance', '<f8')],
      (1.0, 1.0, 0.5), (1.0, 0.5, 0.5)],
     [[(1.4, 1.4, 0.0, 0.0, 100.0),
       (2.0, 0.5, 0.99, 0.99, 150.0)],
      [('mass_1', '<f8'), ('mass_2', '<f8'), ('a_1', '<f8'),
       ('a_2', '<f8'), ('luminosity_distance', '<f8')],
      (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
     [[(1.4, 1.4, 100.0),
       (2.0, 0.5, 150.0)],
      [('mass_1', '<f8'), ('mass_2', '<f8'), ('luminosity_distance', '<f8')],
      (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
     [[(1.4, 1.4, 1.4, 1.4, 0.0, 0.0, 100.0),
       (2.0, 0.5, 2.0, 0.5, 0.99, 0.99, 150.0)],
      [('mass_1_source', '<f8'), ('mass_2_source', '<f8'),
       ('mass_1', '<f8'), ('mass_2', '<f8'), ('a_1', '<f8'),
       ('a_2', '<f8'), ('luminosity_distance', '<f8')],
      (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
     [[(4.5, -0.1, 200.0, 100000, 1.4, 1.4),
       (1.6, 0.3, 201.0, 100000, 1.5, 1.3)],
     [('ra', '<f8'), ('dec', '<f8'), ('luminosity_distance', '<f8'),
      ('time', '<f8'), ('mass_1', '<f8'), ('mass_2', '<f8')],
     (1.0, 1.0, 0.0), (1.0, 1.0, 0.0)],
     [[(4.5, -0.1, 200.0, 100000, 4.5, 4.4),
       (1.6, 0.3, 201.0, 100000, 4.3, 4.2)],
     [('ra', '<f8'), ('dec', '<f8'), ('luminosity_distance', '<f8'),
      ('time', '<f8'), ('mass_1', '<f8'), ('mass_2', '<f8')],
     (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)],
     [[(4.5, -0.1, 200.0, 100000, 40.5, 4.4),
       (1.6, 0.3, 201.0, 100000, 40.3, 4.2)],
     [('ra', '<f8'), ('dec', '<f8'), ('luminosity_distance', '<f8'),
      ('time', '<f8'), ('mass_1', '<f8'), ('mass_2', '<f8')],
     (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]]
)
def test_source_classification_pe(posteriors, dtype, result, result_eos):
    """Test em_bright classification from posterior
    samples - both aligned and precessing cases.
    """
    with NamedTemporaryFile() as f:
        filename = f.name
        with h5py.File(f, 'w') as tmp_h5:
            data = np.array(
                posteriors,
                dtype=dtype
            )
            tmp_h5.create_dataset(
                'posterior_samples',
                data=data
            )
        r = em_bright.source_classification_pe(filename)
        r_eos = em_bright.source_classification_pe(filename, num_eos_draws=5,
                                                   eos_seed=0)
    assert r == result
    assert r_eos == result_eos


@pytest.mark.parametrize(
    'm1,m2,chi1,chi2,snr,result_ns,result_em,result_mg',
    [[1.1, 1.0, 0.0, 0.0, 20.0, 1.0, 1.0, 0.0],
     [1.1, 1.0, 0.0, 0.0, 5.0, 1.0, 1.0, 0.0],
     [100.0, 50.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
     [8.0, 1.4, -0.99, 0.0, 20.0, 1.0, 0.0, 0.0],
     [8.0, 1.4, 0.99, 0.0, 20.0, 1.0, 1.0, 0.04]]
)
def test_source_classification_sanity(m1, m2, chi1, chi2, snr,
                                      result_ns, result_em, result_mg):
    '''Test results for limiting cases'''
    ns, em, mg = em_bright.source_classification(m1, m2, chi1, chi2, snr)
    assert ns == pytest.approx(result_ns, abs=1e-2)
    assert em == pytest.approx(result_em, abs=1e-2)
    assert mg == pytest.approx(result_mg, abs=1e-2)


def test_mock_classifier():
    '''Test to check usage of custom classifier'''
    m1 = 1.1
    m2 = 1.0
    chi1 = 0.04
    chi2 = 0.0
    snr = 20.
    # Define mock objects
    mock_clf_ns = Mock()
    mock_clf_emb = Mock()
    mock_clf_ns.predict_proba = Mock(return_value=np.array([[1., 1.]]))
    mock_clf_emb.predict_proba = Mock(return_value=np.array([[1., 1.]]))

    ns, em, mg = em_bright.source_classification(m1, m2, chi1, chi2, snr,
                                                 ns_classifier=mock_clf_ns,
                                                 emb_classifier=mock_clf_emb)
    mock_clf_ns.predict_proba.assert_called_once()
    mock_clf_emb.predict_proba.assert_called_once()
    assert ns == 1.0
    assert em == 1.0
    assert mg == 0.0


@pytest.mark.parametrize(
    'm1,m2,chi1,chi2,eosname,non_zero_remnant',
    [[1.1, 1.0, 0.0, 0.0, "2H", 1.0],
     [1.1, 1.0, 0.0, 0.0, "SLy", 1.0],
     [1.1, 1.0, 0.0, 0.0, "AP4", 1.0],
     [1.1, 1.0, 0.0, 0.0, "WFF1", 1.0],
     [26.0, 2.6, 0.0, 0.0, "2H", 0.0],
     [10.0, 1.4, 0.99, 0.0, "2H", 1.0],
     [10.0, 1.4, -0.99, 0.0, "2H", 0.0]]
)
def test_compute_disk_mass(m1, m2, chi1, chi2,
                           eosname, non_zero_remnant):
    has_remnant = em_bright.computeDiskMass.computeDiskMass(
        m1, m2, chi1, chi2, eosname=eosname
    ) > 0.
    assert has_remnant == non_zero_remnant


@pytest.mark.parametrize(
    'm1,m2,chi1,chi2,non_zero_remnant',
    [[1.4, 1.4, 0., 0., 1.0],
     [50, 50., 0., 0., 0.]]
)
def test_compute_disk_mass_eos_marginalization(m1, m2, chi1, chi2,
                                               non_zero_remnant):
    np.random.seed(1)
    num_eos_draws = 1
    ALL_EOS_DRAWS = utils.load_eos_posterior()
    rand_subset = np.random.choice(
        len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS))  # noqa:E501
    subset_draws = ALL_EOS_DRAWS[rand_subset]
    M, R = subset_draws['M'], subset_draws['R']
    max_mass = np.max(M)
    mass_radius_relation = interp1d(M[0], R[0], bounds_error=False)
    has_remnant = em_bright.computeDiskMass.computeDiskMass(m1, m2, chi1, chi2, eosname=mass_radius_relation, max_mass=max_mass)  # noqa:E501
    assert has_remnant == non_zero_remnant


@pytest.mark.parametrize(
    'masses_spins',
    [np.array([2.0, 2.0, 0.0, 0.0]),
     np.array([5.0, 2.0, 0.99, 0.0])]
)
def test_compute_disk_mass_numpy_scalar(masses_spins):
    M_remnant_numpy_float = em_bright.computeDiskMass.computeDiskMass(
        *(v for v in masses_spins), eosname="2H"
    )
    M_remnant_float = em_bright.computeDiskMass.computeDiskMass(
        *(float(v) for v in masses_spins), eosname="2H"
    )
    assert M_remnant_numpy_float == M_remnant_float


def test_embright_categorization():
    for eosname, max_mass in EOS_MAX_MASS.items():
        with NamedTemporaryFile() as tf:
            outFile = tf.name
        categorize.embright_categorization(
            Path(__file__).parents[0] / 'data/test_categorize_data.tbl',
            outFile, eosname=eosname
        )
        # read in pickle file
        res = pd.read_pickle(outFile)
        assert ~np.all(
            np.logical_xor(
                res.NS.values.astype(bool),
                res.m2_inj.values < EOS_MAX_MASS[eosname]
            )
        )


def test_mass_gap_categorization():
    with NamedTemporaryFile() as tf:
        outFile = tf.name
    categorize.mass_gap_categorization(
        Path(__file__).parents[0] / 'data/test_categorize_data.tbl',
        outFile
    )
    # read in pickle file
    res = pd.read_pickle(outFile)
    expected_mask = (
        (res.m1_inj_source > 3.0) & (res.m1_inj_source < 5.0)
        |
        (res.m2_inj_source > 3.0) & (res.m2_inj_source < 5.0)
    )
    assert ~np.all(
        np.logical_xor(
            res.mass_gap.values.astype(bool),
            expected_mask
        )
    )
