from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import h5py
import pandas as pd

import pytest
from unittest.mock import Mock

from .. import em_bright, categorize, EOS_MAX_MASS


def test_version():
    from .. import __version__
    assert __version__ == '1.0.3'


@pytest.mark.parametrize(
    'posteriors, dtype, result',
    [[[(1.2, 1.0, 0.0, 0.0, 0.0, 0.0, 100.0),
       (2.0, 0.5, 0.99, 0.99, 0.0, 0.0, 150.0)],
      [('mc', '<f8'), ('q', '<f8'), ('a1', '<f8'),
       ('a2', '<f8'), ('tilt1', '<f8'), ('tilt2', '<f8'),
       ('dist', '<f8')],
      (1.0, 1.0)],
     [[(1.2, 1.0, 0.0, 0.0, 100.0),
       (2.0, 0.5, 0.99, 0.99, 150.0)],
      [('mc', '<f8'), ('q', '<f8'), ('a1', '<f8'),
       ('a2', '<f8'), ('dist', '<f8')],
      (1.0, 1.0)]]
)
def test_source_classification_pe(posteriors, dtype, result):
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
                'lalinference/lalinference_mcmc/posterior_samples',
                data=data
            )
        r = em_bright.source_classification_pe(filename)
    assert r == result


@pytest.mark.parametrize(
    'm1,m2,chi1,chi2,snr,result_ns, result_em',
    [[1.1, 1.0, 0.0, 0.0, 20.0, 1.0, 1.0],
     [1.1, 1.0, 0.0, 0.0, 5.0, 1.0, 1.0],
     [100.0, 50.0, 0.0, 0.0, 10.0, 0.0, 0.0],
     [8.0, 1.4, -0.99, 0.0, 20.0, 1.0, 0.0],
     [8.0, 1.4, 0.99, 0.0, 20.0, 1.0, 1.0]]
)
def test_source_classification_sanity(m1, m2, chi1, chi2, snr,
                                      result_ns, result_em):
    '''Test results for limiting cases'''
    ns, em = em_bright.source_classification(m1, m2, chi1, chi2, snr)
    assert ns == pytest.approx(result_ns)
    assert em == pytest.approx(result_em)


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

    ns, em = em_bright.source_classification(m1, m2, chi1, chi2, snr,
                                             ns_classifier=mock_clf_ns,
                                             emb_classifier=mock_clf_emb)
    mock_clf_ns.predict_proba.assert_called_once()
    mock_clf_emb.predict_proba.assert_called_once()
    assert ns == 1.0
    assert em == 1.0


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
