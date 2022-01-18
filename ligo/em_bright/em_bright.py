"""Module containing tools for EM-Bright classification of
compact binaries using trained supervised classifier
"""
import h5py
import pickle

import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology, units as u

from . import PACKAGE_FILENAMES
from .computeDiskMass import computeDiskMass


def mchirp(m1, m2):
    return(m1 * m2)**(3./5.)/(m1 + m2)**(1./5.)


def q(m1, m2):
    return m2/m1 if m2 < m1 else m1/m2


def source_classification(m1, m2, chi1, chi2, snr,
                          ns_classifier=None,
                          emb_classifier=None):
    """
    Computes ``HasNS`` and ``HasRemnant`` probabilities
    from point mass, spin and signal to noise ratio
    estimates.

    Parameters
    ----------
    m1 : float
        primary mass
    m2 : float
        secondary mass
    chi1 : float
        dimensionless primary spin
    chi2 : float
        dimensionless secondary spin
    snr : float
        signal to noise ratio of the signal
    ns_classifier : object, optional
        pickled object for NS classification
    emb_classifier : object, optional
        pickled object for EM brightness classification

    Returns
    -------
    tuple
        (P_NS, P_EMB) predicted values.

    Notes
    -----
    By default the classifiers are trained using the
    ``KNearestNeighbor`` algorithm from ``scikit-learn``,
    data is used to make predictions. Custom `ns_classifier`,
    `emb_classifier` can be supplied so long as they provide
    ``predict_proba`` method and the feature set is
    [[mass1, mass2, spin1z, spin2z, snr]].

    Examples
    --------
    >>> from ligo.em_bright import em_bright
    >>> em_bright.source_classification(2.0 ,1.0 ,0. ,0. ,10.0)
    (1.0, 1.0)
    """
    if not ns_classifier:
        with open(PACKAGE_FILENAMES['knn_ns_classifier.pkl'], 'rb') as f:
            ns_classifier = pickle.load(f)
    if not emb_classifier:
        with open(PACKAGE_FILENAMES['knn_em_classifier.pkl'], 'rb') as f:
            emb_classifier = pickle.load(f)

    features = [[m1, m2, chi1, chi2, snr]]
    prediction_em, prediction_ns = \
        emb_classifier.predict_proba(features).T[1], \
        ns_classifier.predict_proba(features).T[1]
    return prediction_ns[0], prediction_em[0]


def get_redshifts(distances, N=10000):
    """
    Compute redshift using the Planck15 cosmology.

    Parameters
    ----------
    distances: float or numpy.ndarray
              distance(s) in Mpc

    N : int, optional
      Number of steps for the computation of the interpolation function

    Example
    -------
    >>> distances = np.linspace(10, 100, 10)
    >>> em_bright.get_redshifts(distances)
    array([0.00225566, 0.00450357, 0.00674384, 0.00897655,
           0.01120181, 0.0134197 , 0.01563032, 0.01783375
           0.02003009, 0.02221941])

    Notes
    -----
    This function accepts HDF5 posterior samples file and computes
    redshift by interpolating the distance-redshift relation.
    """
    function = cosmology.Planck15.luminosity_distance
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    z_min = cosmology.z_at_value(func=function, fval=min_dist*u.Mpc)
    z_max = cosmology.z_at_value(func=function, fval=max_dist*u.Mpc)
    z_steps = np.linspace(z_min - (0.1*z_min), z_max + (0.1*z_min), N)
    lum_dists = cosmology.Planck15.luminosity_distance(z_steps)
    s = interp1d(lum_dists, z_steps)
    redshifts = s(distances)
    return redshifts


def source_classification_pe(posterior_samples_file, hdf5=True,
                             threshold=3.0, sourceframe=True):
    """
    Compute ``HasNS`` and ``HasRemnant`` probabilities from posterior
    samples.

    Parameters
    ----------
    posterior_samples_file : str
        Posterior samples file

    hdf5 : bool, optional
        Supply when not using HDF5 format

    threshold : float, optional
        Maximum neutron star mass for `HasNS` computation

    sourceframe : bool, optional
        Supply to use detector frame quantities

    Returns
    -------
    tuple
        (P_NS, P_EMB) predicted values.


    Examples
    --------
    >>> from ligo.em_bright import em_bright
    >>> em_bright.source_classification_pe('posterior_samples.hdf5')
    (1.0, 0.9616727412238634)
    >>> em_bright.source_classification_pe('posterior_samples.dat', hdf5=False)  # noqa:E501
    (0.0, 0.0)
    """
    if hdf5:
        with h5py.File(posterior_samples_file, 'r') as data:
            engine = list(data['lalinference'].keys())[0]
            samples = data['lalinference'][engine]['posterior_samples'][()]
        mc_det_frame = samples['mc']
        lum_dist = samples['dist']
        redshifts = get_redshifts(lum_dist)
        if sourceframe:
            mc = mc_det_frame/(1 + redshifts)
        else:
            mc = mc_det_frame

    else:
        samples = np.recfromtxt(posterior_samples_file, names=True)
        if sourceframe:
            mc = samples['mc_source']
        else:
            mc = samples['mc']

    q = samples['q']
    m1 = mc * (1 + q)**(1/5) * (q)**(-3/5)
    m2 = mc * (1 + q)**(1/5) * (q)**(2/5)

    try:
        chi1 = samples['a1'] * np.cos(samples['tilt1'])
        chi2 = samples['a2'] * np.cos(samples['tilt2'])
    except ValueError:
        # for aligned-spin PE, a1, a2 is the z component
        chi1 = samples['a1']
        chi2 = samples['a2']

    M_rem = computeDiskMass(m1, m2, chi1, chi2)
    prediction_ns = np.sum(m2 <= threshold)/len(m2)
    prediction_em = np.sum(M_rem > 0)/len(M_rem)

    return prediction_ns, prediction_em
