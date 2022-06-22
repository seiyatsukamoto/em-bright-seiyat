"""
Module containing functions to calculate mass ejecta and lightcurves from
initial component masses
"""
import numpy as np
import time
from scipy.interpolate import interpolate as interp
from configparser import ConfigParser
from pathlib import Path

from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.EjectaFits import PaDi2019
from gwemlightcurves.EjectaFits import KrFo2019

rel_path = 'etc/conf.ini'
conf_path = Path(__file__).parents[3] / rel_path
config = ConfigParser()
config.read(conf_path)
fix_seed = config.get('lightcurve_configs', 'fix_seed')
if fix_seed:
    np.random.seed(0)


def initial_mass_draws(dist, mass_draws):
    '''
    Draws component masses (NS's or BH's) from the desired distribution

    Parameters
    ----------
    dist: str
        one of the below component mass dists
    mass_draws: int
        number of component mass pairs to draw

    Returns
    -------
    m1: numpy array
        more massive component mass
    m2: numpy array
        less massive component mass
    merger_type: str
        can be BNS, NSBH, or NotSpecified
    '''

    m1_unsorted, m2_unsorted, merger_type = dist(mass_draws)
    # sort to make sure m1 > m2
    m1 = np.maximum(m1_unsorted, m2_unsorted)
    m2 = np.minimum(m1_unsorted, m2_unsorted)

    return m1, m2, merger_type


def run_EOS(m1, m2, thetas, N_EOS=100, EOS_draws=None):
    '''
    Pick EOS and calculate ejecta quantities, including total mass ejecta,
    dyn and wind ejecta, velocity of ejecta, compactness, tidal deformability

    Parameters
    ----------
    m1: float
        component mass 1
    m2: float
        component mass 2
    thetas: np.array
        array of theta draws
    N_EOS: int
        number of EOS draws
    EOS_draws: np.array
        array of EOS draws

    Returns
    -------
    samples: KNTable object
        ejecta quantities
    '''

    mchirp, eta, q = lightcurve_utils.ms2mc(m1, m2)

    rel_path = 'etc/conf.ini'
    conf_path = Path(__file__).parents[3] / rel_path
    config = ConfigParser()
    config.read(conf_path)
    model_dict = eval(config.get('lightcurve_configs', 'lightcurve_model'))
    model, chi_eff = model_dict['model'], model_dict['chi_eff']

    data = np.vstack((m1, m2, chi_eff, mchirp, eta, q)).T
    samples = KNTable((data),
                      names=('m1', 'm2', 'chi_eff', 'mchirp', 'eta', 'q'))

    samples = EOS_samples(samples, thetas, N_EOS, EOS_draws)
    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)

    # Calc compactness
    samples = samples.calc_compactness(fit=True)

    # Calc baryonic mass
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)
    samples['merger_type'] = ''

    # 1 is BNS, 2 is NSBH, 3 is BBH
    idx1 = np.where((samples['m1'] <= samples['mbns']) &
                    (samples['m2'] <= samples['mbns']))[0]
    idx2 = np.where((samples['m1'] > samples['mbns']) &
                    (samples['m2'] <= samples['mbns']))[0]
    idx3 = np.where((samples['m1'] > samples['mbns']) &
                    (samples['m2'] > samples['mbns']))[0]

    # BNS
    samples['merger_type'][idx1] = 1
    # NSBH
    samples['merger_type'][idx2] = 2
    # BBH
    samples['merger_type'][idx3] = 3

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
    err_dict = eval(config.get('lightcurve_configs','mej_error_dict'))
    error = err_dict['error']
    if error == 'log':
        samples['mej'] = np.power(10.,
                                  np.random.normal(np.log10(samples['mej']),
                                  err_dict['log_val']))
    elif error == 'lin':
        samples['mej'] = np.random.normal(samples['mej'],
                                          err_dict['lin_val']*samples['mej'])
    elif error == 'loggauss':
        samples['mej'] = np.power(10.,
                                  np.random.normal(np.log10(samples['mej']),
                                                   err_dict['loggauss_val']))

    # TEMPORARY, remove once entire EOS file is implemented
    print('min mej:')
    print(np.min(samples['mej']))
    if np.min(samples['mej']) < 0:
        print('---------------mej less than zero!!!-----------------')
    idx = np.where(samples['mej'] <= 0)[0]
    samples['mej'][idx] = 1e-11

    if (model == 'Bu2019inc'):
        idx = np.where(samples['mej'] <= 1e-6)[0]
        samples['mej'][idx] = 1e-11
    elif (model == 'Ka2017'):
        idx = np.where(samples['mej'] <= 1e-3)[0]
        samples['mej'][idx] = 1e-11

    return samples


def EOS_samples(samples, thetas, nsamples, EOS_draws):
    '''
    Draws different EOS's for ejecta calculations

    Parameters
    ----------
    samples: KNTable object
        table of ejecta quantities
    thetas: np.array
        array of theta draws
    nsamples: int
        number of EOS draws
    EOS_posterior: str
        filename of posterior samples
    EOS_draws: KNTable object
        EOS draws to be used

    Returns
    -------
    samples: KNTable object
        table of ejecta quantities
    '''  

    rel_path = 'etc/conf.ini'
    conf_path = Path(__file__).parents[3] / rel_path
    config = ConfigParser()
    config.read(conf_path)
    model_dict = eval(config.get('lightcurve_configs', 'lightcurve_model'))
    chi = model_dict['chi_eff']
    

    lambda1s, lambda2s, m1s, m2s = [], [], [], []
    chi_effs, mbnss, qs, mchirps, etas = [], [], [], [], []

    EOS_metadata = {'fix_seed': config.get('lightcurve_configs', 'fix_seed')}
    meta_indices = []    

    # read Phil + Reed's EOS files
    for ii, row in enumerate(samples):
        indices = np.random.choice(len(EOS_draws), size=nsamples)
        meta_indices.append(indices)
        print('idx: '+str(indices))
        for index in indices:
            lambda1, lambda2 = -1, -1
            mbns = -1
            # samples lambda's from Phil + Reed's files
            while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                phasetr = 0
                data_out = EOS_draws[index] 
                marray, larray = data_out['M'], data_out['Lambda']
                f = interp.interp1d(marray, larray, 
                                    fill_value=0, bounds_error=False)
                # pick lambda from least compact stable branch
                if float(f(row['m1'])) > lambda1: lambda1 = f(row['m1'])
                if float(f(row['m2'])) > lambda2: lambda2 = f(row['m2'])
                # get global maximum mass
                if np.max(marray) > mbns: mbns = np.max(marray)
                # check all stable branches
                phasetr += 1

                if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                    # pick a different EOS if it returns 
                    # negative Lambda or Mmax
                    index = int(np.random.choice(len(EOS_draws), size=1))
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

    EOS_metadata['m1s'], EOS_metadata['m2s'] = m1s, m2s
    EOS_metadata['indices'] = indices

    mej_err = eval(config.get('lightcurve_configs', 'mej_error_dict'))
    EOS_metadata['mej_err'] = mej_err
    date_time = time.strftime('%Y%m%d-%H%M%S')

    filename = f'etc/metadata/EOS/EOS_meta_{date_time}.txt'
    folder_path = Path(__file__).parents[3]
    file_path = folder_path / filename
    Path(folder_path / 'etc/metadata/EOS').mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(str(EOS_metadata))

    thetas[thetas > 90] = 180 - thetas[thetas > 90]

    # create a new table including each EOS draw for each component mass pair, and new quantities
    data = np.vstack((m1s, m2s, lambda1s, lambda2s, chi_effs, thetas, mbnss, qs, mchirps, etas)).T
    samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2', 'chi_eff', 'theta', 'mbns', 'q', 'mchirp', 'eta'))

    return samples

def ejecta_to_lc(samples, save_pkl = False): #maybe reimplement saving to a pkl? Depends on final low-latency product
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
    lightcurve_data: dictionary
        ejecta quatities and lightcurves for various mag bands
    '''

    num_samples = len(samples)
    phis = 45 * np.ones(num_samples)
    samples['phi'] = phis

    rel_path = 'etc/conf.ini'
    conf_path = Path(__file__).parents[3] / rel_path
    config = ConfigParser()
    config.read(conf_path)

    # intitial time, final time, and timestep for lightcurve calculation
    model_dict = eval(config.get('lightcurve_configs', 'lightcurve_model'))
    samples['tini'], samples['tmax'], samples['dt'] = model_dict['tini'], model_dict['tmax'], model_dict['dt']

    kwargs = eval(config.get('lightcurve_configs', 'kwargs'))
    svd_path = 'svdmodels'
    model_path = Path(__file__).parents[0] / svd_path
    kwargs['ModelPath'] = model_path

    model = model_dict['model']

    LC_metadata = {'model_conf': model_dict}
    date_time = time.strftime('%Y%m%d-%H%M%S')

    filename = f'etc/metadata/LC/LC_meta_{date_time}.txt'
    folder_path = Path(__file__).parents[3]
    file_path = folder_path / filename

    Path(folder_path / 'etc/metadata/LC').mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        f.write(str(LC_metadata))

    lightcurve_data = KNTable.model(model, samples, **kwargs)

    return lightcurve_data
