"""
Module containing functions to calculate mass ejecta and lightcurves from
initial component masses
"""
import numpy as np
import astropy
import pickle
import time
from scipy.interpolate import interpolate as interp
from configparser import ConfigParser
from pathlib import Path

import ligo.em_bright.lightcurves.lightcurve_utils as em_bright
from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.EjectaFits import PaDi2019
from gwemlightcurves.EjectaFits import KrFo2019

# load configs
rel_path = 'etc/conf.ini'
conf_path = Path(__file__).parents[3] / rel_path
config = ConfigParser()
config.read(conf_path)
fix_seed = config.get('lightcurve_configs', 'fix_seed')
if fix_seed:
    np.random.seed(0)

# load posterior
draws = load_EOS_posterior()
# load lightcurve model
model_dict = eval(config.get('lightcurve_configs', 'lightcurve_model'))
kwargs = eval(config.get('lightcurve_configs', 'kwargs'))
svd_path = 'data'
model_path = Path(__file__).parents[1] / svd_path
kwargs['ModelPath'] = model_path
model = model_dict['model']
mag_model = model + '_mag.pkl'
lbol_model = model + '_lbol.pkl'
with open(model_path / mag_model, 'rb') as f:
    svd_mag_model = pickle.load(f)
with open(model_path / lbol_model, 'rb') as f:
    svd_lbol_model = pickle.load(f)
# time for meta data
date_time = time.strftime('%Y%m%d-%H%M%S')


def lightcurve_predictions(m1s=None, m2s=None, thetas=None,
                           mass_dist=None, mass_draws=None, N_eos=50):
    '''
    Main function to carry out ejecta quantity and lightcurve
    predictions. Needs either: m1, m2, and theta OR mass_dist

    Parameters
    ----------
    m1s: numpy array
        more massive component masses in solar masses
    m2s: numpy array
        less massive component masses in solar masses
    thetas: numpy array
        inclination angles in radians
    mass_dist: str
        one of the mass dists found in mass_distributions
    mass_draws: int
        number of masses to draw if using mass_dist
    N_eos: int
        number of eos draws

    Returns
    -------
    lightcurve_data: astropy table object
        ejecta quantities and lightcurves for various mag bands
    has_Remnant: float
        fraction (0 to 1) of mergers with ejecta mass > 1e-3 solar masses
    eos_metadata: astropy table object
        meta data describing eos draws
    lightcurve_metadata: astropy table object
        meta data describing lightcurve calculation
    '''

    # draw masses from dist
    if mass_dist:
        m1s, m2s = initial_mass_draws(mass_dist, mass_draws)
        thetas = 180. * np.arccos(np.random.uniform(-1., 1., len(m1s))) / np.pi
        idx_thetas = np.where(thetas > 90.)[0]
        thetas[idx_thetas] = 180. - thetas[idx_thetas]

    lightcurve_data = []
    all_eos_metadata = []
    for i, m1 in enumerate(m1s):
        samples, eos_metadata = run_eos(m1, m2s[i], thetas[i],
                                        N_eos=N_eos, eos_draws=draws)
        lightcurves, lightcurve_metadata = ejecta_to_lightcurve(samples)
        lightcurve_data.append(lightcurves)
        all_eos_metadata.append(eos_metadata)
    lightcurve_data = astropy.table.vstack(lightcurve_data)
    eos_metadata = astropy.table.vstack(all_eos_metadata)

    remnant = lightcurve_data[lightcurve_data['mej'] > 1e-3]
    has_Remnant = len(remnant)/len(lightcurve_data['mej'])

    return lightcurve_data, has_Remnant, eos_metadata, lightcurve_metadata


def initial_mass_draws(dist, mass_draws):
    '''
    Draws component masses (NS's or BH's) from the desired distribution

    Parameters
    ----------
    dist: str
        one of the mass dists found in mass_distributions
    mass_draws: int
        number of component mass pairs to draw

    Returns
    -------
    m1: numpy array
        more massive component mass in solar masses
    m2: numpy array
        less massive component mass in solar masses
    '''

    m1_unsorted, m2_unsorted, merger_type = dist(mass_draws)
    # sort to make sure m1 > m2
    m1 = np.maximum(m1_unsorted, m2_unsorted)
    m2 = np.minimum(m1_unsorted, m2_unsorted)

    return m1, m2


def run_eos(m1, m2, thetas, N_eos=50, eos_draws=None):
    '''
    Uses eos draws provided and calculates ejecta quantities, including
    total mass ejecta, dyn and wind ejecta, velocity of ejecta,
    compactness, and tidal deformability

    Parameters
    ----------
    m1: float
        component mass 1
    m2: float
        component mass 2
    thetas: np.array
        array of theta draws
    N_eos: int
        number of eos draws
    eos_draws: np.array
        array of eos draws

    Returns
    -------
    samples: KNTable object
        ejecta quantities
    eos_metadata: astropy table object
        meta data describing eos draws
    '''

    mchirp, eta, q = lightcurve_utils.ms2mc(m1, m2)
    model, chi_eff = model_dict['model'], model_dict['chi_eff']

    data = np.vstack((m1, m2, chi_eff, mchirp, eta, q)).T
    samples = KNTable(data,
                      names=('m1', 'm2', 'chi_eff', 'mchirp', 'eta', 'q'))

    samples, eos_metadata = eos_samples(samples, thetas, N_eos, eos_draws)
    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)

    # Calc compactness
    samples = samples.calc_compactness(fit=True)

    # Calc baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    samples['merger_type'] = len(samples)*['undefined']
    samples['merger_type'].astype('str')

    # 1 is BNS, 2 is NSBH, 3 is BBH
    idx1 = np.where((samples['m1'] <= samples['mbns']) &
                    (samples['m2'] <= samples['mbns']))[0]
    idx2 = np.where((samples['m1'] > samples['mbns']) &
                    (samples['m2'] <= samples['mbns']))[0]
    idx3 = np.where((samples['m1'] > samples['mbns']) &
                    (samples['m2'] > samples['mbns']))[0]

    # include merger type
    samples['merger_type'][idx1] = 'BNS'
    samples['merger_type'][idx2] = 'NSBH'
    samples['merger_type'][idx3] = 'BBH'

    mej = np.zeros(samples['m1'].shape)
    vej = np.zeros(samples['m1'].shape)
    wind_mej = np.zeros(samples['m1'].shape)
    dyn_mej = np.zeros(samples['m1'].shape)

    # calc the mass of ejecta
    mej1, dyn_mej1, wind_mej1 = PaDi2019.calc_meje(samples['m1'],
                                                   samples['c1'],
                                                   samples['m2'],
                                                   samples['c2'],
                                                   split_mej=True)

    # calc the velocity of ejecta
    vej1 = PaDi2019.calc_vej(samples['m1'], samples['c1'],
                             samples['m2'], samples['c2'])

    # calc the mass of ejecta
    mej2, dyn_mej2, wind_mej2 = KrFo2019.calc_meje(samples['q'],
                                                   samples['chi_eff'],
                                                   samples['c2'],
                                                   samples['m2'],
                                                   split_mej=True)

    # calc the velocity of ejecta
    vej2 = KrFo2019.calc_vave(samples['q'])

    # calc the mass of ejecta
    mej3 = np.zeros(samples['m1'].shape)
    dyn_mej3 = np.zeros(samples['m1'].shape)
    wind_mej3 = np.zeros(samples['m1'].shape)
    # calc the velocity of ejecta
    vej3 = np.zeros(samples['m1'].shape) + 0.2

    mej[idx1], vej[idx1] = mej1[idx1], vej1[idx1]
    mej[idx2], vej[idx2] = mej2[idx2], vej2[idx2]
    mej[idx3], vej[idx3] = mej3[idx3], vej3[idx3]

    wind_mej[idx1], dyn_mej[idx1] = wind_mej1[idx1], dyn_mej1[idx1]
    wind_mej[idx2], dyn_mej[idx2] = wind_mej2[idx2], dyn_mej2[idx2]
    wind_mej[idx3], dyn_mej[idx3] = wind_mej3[idx3], dyn_mej3[idx3]

    samples['mej'] = mej
    samples['vej'] = vej
    samples['dyn_mej'] = dyn_mej
    samples['wind_mej'] = wind_mej

    # Add draw from a gaussian in the log of
    # ejecta mass with 1-sigma size of 70%
    err_dict = eval(config.get('lightcurve_configs', 'mej_error_dict'))
    error = err_dict['error']
    if error == 'log':
        samples['mej'] = np.power(10., np.random.normal(np.log10(samples['mej']), err_dict['log_val']))  # noqa:E501
    elif error == 'lin':
        samples['mej'] = np.random.normal(samples['mej'], err_dict['lin_val']*samples['mej'])  # noqa:E501
    elif error == 'loggauss':
        samples['mej'] = np.power(10., np.random.normal(np.log10(samples['mej']), err_dict['loggauss_val']))  # noqa:E501

    if np.min(samples['mej']) < 0:
        print('---------------mej less than zero!!!-----------------')
    idx = np.where(samples['mej'] <= 0)[0]
    samples['mej'][idx] = err_dict['min_mej']

    if (model == 'Bu2019inc'):
        idx = np.where(samples['mej'] <= 1e-6)[0]
        samples['mej'][idx] = 1e-11
    elif (model == 'Ka2017'):
        idx = np.where(samples['mej'] <= 1e-3)[0]
        samples['mej'][idx] = 1e-11

    return samples, eos_metadata


def eos_samples(samples, thetas, nsamples, eos_draws):
    '''
    Draws different eos's for ejecta calculations

    Parameters
    ----------
    samples: KNTable object
        table of ejecta quantities
    thetas: np.array
        array of theta draws
    nsamples: int
        number of eos draws
    eos_draws: KNTable object
        eos draws to be used

    Returns
    -------
    samples: KNTable object
        table of ejecta quantities
    eos_metadata: astropy table object
        meta data describing eos draws
    '''

    lambda1s, lambda2s, m1s, m2s = [], [], [], []
    chi_effs, mbnss, qs, mchirps, etas = [], [], [], [], []
    # duplicate thetas for each eos draw
    thetas = thetas*np.ones(nsamples)

    eos_metadata = {'fix_seed': config.get('lightcurve_configs', 'fix_seed')}
    meta_indices = []

    # read Phil + Reed's eos files
    for ii, row in enumerate(samples):
        indices = np.random.choice(len(eos_draws), size=nsamples)
        meta_indices.append(indices)
        for index in indices:
            lambda1, lambda2 = -1, -1
            mbns = -1
            # samples lambda's from Phil + Reed's files
            while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                phasetr = 0
                data_out = eos_draws[index]
                marray, larray = data_out['M'], data_out['Lambda']
                f = interp.interp1d(marray, larray,
                                    fill_value=0, bounds_error=False)
                # pick lambda from least compact stable branch
                if float(f(row['m1'])) > lambda1:
                    lambda1 = f(row['m1'])
                if float(f(row['m2'])) > lambda2:
                    lambda2 = f(row['m2'])
                # get global maximum mass
                if np.max(marray) > mbns:
                    mbns = np.max(marray)
                # check all stable branches
                phasetr += 1

                if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                    # pick a different eos if it returns
                    # negative Lambda or Mmax
                    index = int(np.random.choice(len(eos_draws), size=1))
                    lambda1, lambda2 = -1, -1
                    mbns = -1

            m1s.append(row['m1'])
            m2s.append(row['m2'])
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
            chi_effs.append(row['chi_eff'])
            mbnss.append(mbns)
            qs.append(row['q'])
            mchirps.append(row['mchirp'])
            etas.append(row['eta'])

    eos_metadata['m1s'], eos_metadata['m2s'] = m1s, m2s
    eos_metadata['indices'] = indices

    mej_err = eval(config.get('lightcurve_configs', 'mej_error_dict'))
    eos_metadata['mej_err'] = mej_err

    # create a new table including each eos draw for each
    # component mass pair, and new quantities
    data = np.vstack((m1s, m2s, lambda1s, lambda2s, chi_effs,
                      thetas, mbnss, qs, mchirps, etas)).T
    samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2', 'chi_eff',
                                   'theta', 'mbns', 'q', 'mchirp', 'eta'))

    return samples, eos_metadata


def ejecta_to_lightcurve(samples):
    '''
    Calculate lightcurves from ejecta quatities

    Parameters
    ----------
    samples: astropy Table
        Table of ejecta quatities
    save_pkl: bool
        whether or not to save lightcurves to pickle files

    Returns
    -------
    lightcurve_data: KNTable object
        ejecta quatities and lightcurves for various mag bands
    lightcurve_metadata: astropy table object
        meta data describing lightcurve calculation
    '''

    num_samples = len(samples)
    phis = 45 * np.ones(num_samples)
    samples['phi'] = phis

    # intitial time, final time, and timestep for lightcurve calculation
    model_dict = eval(config.get('lightcurve_configs', 'lightcurve_model'))
    samples['tini'] = model_dict['tini']
    samples['tmax'] = model_dict['tmax']
    samples['dt'] = model_dict['dt']

    # read from config file
    kwargs['ModelPath'] = model_path
    kwargs['ModelFileMag'] = svd_mag_model
    kwargs['ModelFileLbol'] = svd_lbol_model

    model = model_dict['model']
    lightcurve_metadata = {'model_conf': model_dict}
    lightcurve_data = KNTable.model(model, samples, **kwargs)

    return lightcurve_data, lightcurve_metadata
