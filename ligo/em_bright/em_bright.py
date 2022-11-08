"""Module containing tools for EM-Bright classification of
compact binaries using trained supervised classifier
"""
import h5py
import numpy as np
from scipy.interpolate import interp1d
from astropy import cosmology, units as u

from . import (
    EOS_BAYES_FACTORS,
    PACKAGE_FILENAMES,
    computeDiskMass,
    utils
)

ALL_EOS_DRAWS = utils.load_eos_posterior()

_classifiers = {
    eos: utils._open_and_return_clfs(
        PACKAGE_FILENAMES[f'{eos}.pickle']
    )
    for eos in EOS_BAYES_FACTORS
}
"""Classifiers keyed on EOS names. Order (clf_ns, clf_em)"""
assert set(_classifiers) == set(EOS_BAYES_FACTORS), "Inconsistency in"
" number of trained classifiers."

_massgap_classifier = utils._open_and_return_clfs(
        PACKAGE_FILENAMES['MASS_GAP.pickle']
    )


def mchirp(mass_1, mass_2):
    return (mass_1 * mass_2)**(3./5.)/(mass_1 + mass_2)**(1./5.)


def q(mass_1, mass_2):
    return mass_2/mass_1 if mass_2 < mass_1 else mass_1/mass_2


def source_classification(mass_1, mass_2, chi1, chi2, snr,
                          ns_classifier=None,
                          emb_classifier=None,
                          massgap_classifier=None):
    """
    Computes ``HasNS``, ``HasRemnant``, and ``MassGap``
    probabilities from point mass, spin and
    signal to noise ratio estimates.

    Parameters
    ----------
    mass_1 : float
        primary mass
    mass_2 : float
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
    massgap_classifier : object, optional
        pickled object for EM brightness classification

    Returns
    -------
    tuple
        (HasNS, HasRemnant, HasMassGap) predicted values.
    Notes
    -----
    By default the classifiers, trained based
    on different nuclear equations of state (EoSs)
    are downloaded from the project page:
    https://git.ligo.org/emfollow/em-properties/em-bright.
    The methodology is described in arXiv:1911.00116.
    The score from each classifier is weighted based on
    the bayes factors of individual EoSs as mentioned in
    Table I of arXiv:2104.08681.
    However, if the trained classifiers are supplied
    via ``ns_classifier`` and ``emb_classifier``,
    the score is reported based on the classifier instead
    of re-weighting the score.
    Examples
    --------
    >>> from ligo.em_bright import em_bright
    >>> em_bright.source_classification(2.0 ,1.0 ,0. ,0. ,10.0)
    (1.0, 1.0, 0.0)
    """
    features = [[mass_1, mass_2, chi1, chi2, snr]]
    try:
        # custom classifiers supplied
        return (
            ns_classifier.predict_proba(features).T[1][0],
            emb_classifier.predict_proba(features).T[1][0],
            massgap_classifier.predict_proba(features).T[1][0]
        )
    except AttributeError as e:
        msg, *_ = e.args
        if msg != """'NoneType' object has no attribute 'predict_proba'""":
            raise

    reweighted_ns_score = reweighted_emb_score = 0.
    for eosname, bayes_factor in EOS_BAYES_FACTORS.items():
        ns_classifier, emb_classifier = _classifiers[eosname]
        reweighted_ns_score += ns_classifier.predict_proba(
            features).T[1][0] * bayes_factor
        reweighted_emb_score += emb_classifier.predict_proba(
            features).T[1][0] * bayes_factor
    massgap_score = _massgap_classifier.predict_proba(features).T[1][0]
    return reweighted_ns_score, reweighted_emb_score, massgap_score


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
    z_steps = np.linspace(z_min - (0.1*z_min), z_max + (0.1*z_max), N)
    lum_dists = cosmology.Planck15.luminosity_distance(z_steps)
    s = interp1d(lum_dists, z_steps)
    redshifts = s(distances)
    return redshifts


def source_classification_pe(posterior_samples_file, threshold=3.0,
                             num_eos_draws=None, eos_seed=None):
    """
    Compute ``HasNS``, ``HasRemnant``, and ``HasMassGap`` probabilities
    from posterior samples.

    Parameters
    ----------
    posterior_samples_file : str
        Posterior samples file

    threshold : float, optional
        Maximum neutron star mass for `HasNS` computation

    num_eos_draws : int
        providing an int here runs eos marginalization
        with the value determining how many eos's to draw

    eos_seed : int
        seed for random eos draws

    Returns
    -------
    tuple
        (HasNS, HasRemnant, HasMassGap) predicted values.


    Examples
    --------
    >>> from ligo.em_bright import em_bright
    >>> em_bright.source_classification_pe('posterior_samples.hdf5')
    (1.0, 0.96, 0.0)
    """
    with h5py.File(posterior_samples_file, 'r') as data:
        samples = data['posterior_samples'][()]

    try:
        mass_1, mass_2 = samples['mass_1_source'], samples['mass_2_source']
    except ValueError:
        lum_dist = samples['luminosity_distance']
        redshifts = get_redshifts(lum_dist)
        try:
            mass_1, mass_2 = samples['mass_1'], samples['mass_2']
            mass_1, mass_2 = mass_1/(1 + redshifts), mass_2/(1 + redshifts)
        except ValueError:
            chirp_mass, mass_ratio = samples['chirp_mass'], samples['mass_ratio']  # noqa:E501
            chirp_mass = chirp_mass/(1 + redshifts)
            mass_1 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(-3/5)  # noqa:E501
            mass_2 = chirp_mass * (1 + mass_ratio)**(1/5) * (mass_ratio)**(2/5)

    try:
        a_1 = samples['a_1'] * np.cos(samples['tilt_1'])
        a_2 = samples['a_2'] * np.cos(samples['tilt_2'])
    except ValueError:
        a_1, a_2 = np.zeros(len(mass_1)), np.zeros(len(mass_2))

    if num_eos_draws:
        np.random.seed(eos_seed)
        prediction_nss, prediction_ems = [], []
        for m1, m2, a1, a2 in zip(mass_1, mass_2, a_1, a_2):
            rand_subset = np.random.choice(
                len(ALL_EOS_DRAWS), num_eos_draws if num_eos_draws < len(ALL_EOS_DRAWS) else len(ALL_EOS_DRAWS))  # noqa:E501
            subset_draws = ALL_EOS_DRAWS[rand_subset]
            M, R = subset_draws['M'], subset_draws['R']
            max_masses = np.max(M, axis=1)
            f_M = [interp1d(m, r, bounds_error=False) for m, r in zip(M, R)]
            for mass_radius_relation, max_mass in zip(f_M, max_masses):
                M_rem = computeDiskMass.computeDiskMass(m1, m2, a1, a2, eosname=mass_radius_relation, max_mass=max_mass)  # noqa:E501
                prediction_nss.append(1. if m2 <= max_mass else 0.)
                prediction_ems.append(1. if M_rem > 0 else 0.)

        prediction_ns = np.mean(prediction_nss)
        prediction_em = np.mean(prediction_ems)

    else:
        M_rem = computeDiskMass.computeDiskMass(mass_1, mass_2, a_1, a_2)
        prediction_ns = np.sum(mass_2 <= threshold)/len(mass_2)
        prediction_em = np.sum(M_rem > 0)/len(M_rem)

    prediction_mg = (mass_1 <= 5) & (mass_1 >= 3)
    prediction_mg += (mass_2 <= 5) & (mass_2 >= 3)
    prediction_mg = np.sum(prediction_mg)/len(mass_1)

    return prediction_ns, prediction_em, prediction_mg
