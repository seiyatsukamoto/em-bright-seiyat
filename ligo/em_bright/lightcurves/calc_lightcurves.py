"""Module containing functions to calculate mass ejecta and lightcurves from
   initial component masses
"""
import numpy as np
from scipy.interpolate import interpolate as interp
from configparser import ConfigParser
from pathlib import Path

from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from gwemlightcurves.EjectaFits import PaDi2019
from gwemlightcurves.EjectaFits import KrFo2019

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


def run_EOS(EOS, m1, m2, thetas, N_EOS=100, EOS_posterior=None, EOS_draws=None, EOS_idx=None):
    '''
    Pick EOS and calculate ejecta quantities, including total mass ejecta, dyn and wind ejecta,
    velocity of ejecta, compactness, tidal deformability 

    Parameters
    ----------
    EOS: str
        equation of state
    m1: float
        component mass 1
    m2: float
        component mass 2
    thetas: np.array
        array of theta draws
    N_EOS: int
        number of EOS draws
    EOS_posterior: str
        filename of posterior samples
    EOS_draws: str
        filenames of EOS draws 
    EOS_idx: str
        indices of EOS draws

    Returns
    -------
    samples: KNTable object
        ejecta quantities
    '''

    q = m2/m1
    mchirp = (m1*m2)**(3/5) / (m1+m2)**(1/5)
    eta = m1*m2/((m1+m2)*(m1+m2))

    rel_path = 'etc/conf.ini'
    conf_path = Path(__file__).parents[3] / rel_path
    config = ConfigParser()
    config.read(conf_path)
    model_dict = eval(config.get('lightcurve_configs', 'Bu2019inc_model'))
    model, chi = model_dict['model'], model_dict['chi']

    data = np.vstack((m1,m2,chi,mchirp,eta,q)).T
    samples = KNTable((data), names = ('m1', 'm2', 'chi_eff', 'mchirp', 'eta', 'q'))
    samples = EOS_samples(samples, thetas, N_EOS, EOS_posterior, EOS_draws, EOS_idx)

    samples = samples.calc_tidal_lambda(remove_negative_lambda=True)

    # Calc compactness
    samples = samples.calc_compactness(fit=True)

    # Calc baryonic mass 
    samples = samples.calc_baryonic_mass(EOS=None, TOV=None, fit=True)

    #----------------------------------------------------------------------------------
    if (not 'mej' in samples.colnames) and (not 'vej' in samples.colnames):
        #1 is BNS, 2 is NSBH, 3 is BBH    
        idx1 = np.where((samples['m1'] <= samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx2 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] <= samples['mbns']))[0]
        idx3 = np.where((samples['m1'] > samples['mbns']) & (samples['m2'] > samples['mbns']))[0]

        mej, vej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)
        wind_mej, dyn_mej = np.zeros(samples['m1'].shape), np.zeros(samples['m1'].shape)   

        # calc the mass of ejecta
        mej1, dyn_mej1, wind_mej1 = PaDi2019.calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'], split_mej=True)
        # calc the velocity of ejecta
        vej1 = PaDi2019.calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
        samples['mchirp'], samples['eta'], samples['q'] = lightcurve_utils.ms2mc(samples['m1'], samples['m2'])

        # calc the mass of ejecta
        mej2, dyn_mej2, wind_mej2 = KrFo2019.calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'], split_mej=True)
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

        # Add draw from a gaussian in the log of ejecta mass with 1-sigma size of 70%
        error = config.get('lightcurve_configs','error')
        if error == 'log':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.236))
        elif error == 'lin':
            samples['mej'] = np.random.normal(samples['mej'],0.72*samples['mej'])
        elif error == 'loggauss':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.312))

        idx = np.where(samples['mej'] <= 0)[0]
        samples['mej'][idx] = 1e-11

        if (model == "Bu2019inc"):
                idx = np.where(samples['mej'] <= 1e-6)[0]
                samples['mej'][idx] = 1e-11
        elif (model == "Ka2017"):
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
    EOS_draws: str
        filenames of EOS draws
    EOS_idx: str
        indices of EOS draws
    EOS: str
        EOS of state used 

    Returns
    -------
    samples: KNTable object
        table of ejecta quantities
    '''  

    rel_path = 'etc/conf.ini'
    conf_path = Path(__file__).parents[3] / rel_path
    config = ConfigParser()
    config.read(conf_path)
    model_dict = eval(config.get('lightcurve_configs', 'Bu2019inc_model'))
    Xlan, chi = model_dict['Xlan'], model_dict['chi']

    lambda1s, lambda2s, m1s, m2s = [], [], [], []
    chi_effs, Xlans, qs, mbnss = [], [], [], []

    m1s, m2s, dists_mbta = [], [], []

    # read Phil + Reed's EOS files
    # idxs = np.array(EOS_posterior["eos"])
    # weights = np.array([np.exp(weight) for weight in EOS_posterior["logweight_total"]])

    for ii, row in enumerate(samples):
        m1, m2, chi_eff = row["m1"], row["m2"], row["chi_eff"]
        # Note: fix weights
        # indices = np.random.choice(np.array(EOS_idx), size=nsamples, replace=True)
        for index in range(nsamples):
            #index = gp10_idx[jj]
            #index = jj 
            lambda1, lambda2 = -1, -1
            mbns = -1
            #data_out = EOS_draws[index]
            # samples lambda's from Phil + Reed's files
            while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                phasetr = 0
                data_out = EOS_draws[index] 
                marray, larray = data_out["M"], data_out["Lambda"]
                f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                if float(f(m1)) > lambda1: lambda1 = f(m1) # pick lambda from least compact stable branch
                if float(f(m2)) > lambda2: lambda2 = f(m2)
                if np.max(marray) > mbns: mbns = np.max(marray) # get global maximum mass

                phasetr += 1 # check all stable branches
                # eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)
                #eospath = "/home/philippe.landry/nseos/eos/gp/mrgagn/DRAWmod1000-%06d/MACROdraw-%06d/MACROdraw-%06d-%d.csv" % (idxs[index]/1000, idxs[index], idxs[index], phasetr)

            if (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                index = int(np.random.choice(np.arange(0,len(idxs)), size=1,replace=True,p=weights/np.sum(weights))) # pick a different EOS if it returns negative Lambda or Mmax
                lambda1, lambda2 = -1, -1
                mbns = -1

            m1s.append(m1)
            m2s.append(m2)
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
            chi_effs.append(chi)
            Xlans.append(Xlan)
            mbnss.append(mbns)

    thetas[thetas > 90] = 180 - thetas[thetas > 90]
    Xlans = np.ones(np.array(m1s).shape) * Xlan

    data = np.vstack((m1s, m2s, lambda1s, lambda2s, Xlans, chi_effs, thetas, mbnss)).T
    samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2', 'Xlan', 'chi_eff', 'theta', 'mbns'))

    return samples

def ejecta_to_lc(samples, save_pkl = False):
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
    model_dict = eval(config.get('lightcurve_configs', 'Bu2019inc_model'))
    samples['tini'], samples['tmax'], samples['dt'] = model_dict['tini'], model_dict['tmax'], model_dict['dt']

    rel_path = 'svdmodels'
    ModelPath = Path(__file__).parents[0] / rel_path
    #ModelPath = "/home/cosmin.stachie/gwemlightcurves/output/svdmodels"
    kwargs = {'SaveModel':False,'LoadModel':True,'ModelPath':ModelPath}
    kwargs["doAB"] = True
    kwargs["doSpec"] = False

    #model = config.get('gwlc_configs','gwlc_Bu2019inc')
    model = model_dict['model']
    model_tables = {}

    sample_split = []

    lightcurve_data = KNTable.model(model, samples, **kwargs)

    return lightcurve_data
