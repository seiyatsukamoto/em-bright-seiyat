"""Module containing functions to calculate mass ejecta and lightcurves from
   initial component masses
"""
import numpy as np
import scipy.stats as stats
from scipy.interpolate import interpolate as interp
from gwemlightcurves import lightcurve_utils
from gwemlightcurves.KNModels import KNTable
from .gwlc_utils import alsing_dist, farrow_dist, zhu_dist

np.random.seed(0)


def initial_mass_draws(Type, mass_draws):
    '''Draws component masses (NS's or BH's) from the desired distribution

       Parameters
       ----------
       Type: str
           one of the below component mass dists
       mass_draws: int
           number of component mass pairs to draw

       Returns
       -------
       m1_sorted: numpy array
           more massive component mass
       m2_sorted: numpy array
           less massive component mass
       thetas: numpy array
           inclination or viewing angle
       MergerType: str
           can be BNS, NSBH, or NotSpecified
    '''

    if Type == 'NSBH_uniform':
        m1 = 3*np.ones(mass_draws)+5*np.random.rand(mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'NSBH'

    elif Type == 'BNS_uniform':
        m1 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'BNS'

    elif Type == 'BNS_alsing':
        # From Alsing 2018 (arxiv 1709.07889)
        a_dist = alsing_dist(a=1.1, b=2.8)
        m1 = a_dist.rvs(size=mass_draws)
        m2 = a_dist.rvs(size=mass_draws)
        MergerType = 'BNS'

    elif Type == 'BNS_farrow':
        # From Farrow 2019 (arxiv 1902.03300)
        f_dist = farrow_dist(a=1.1, b=2.8)
        m1 = f_dist.rvs(size=mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'BNS'

    elif Type == 'NSBH_zhu':
        # From Zhu 2021 (arxiv 2011.02717)
        z_dist = zhu_dist(a=2.8, b=25)
        dist_NS_zhu = stats.norm(1.33, scale=.01)
        m1 = z_dist.rvs(size=mass_draws)
        m2 = dist_NS_zhu.rvs(size=mass_draws)
        MergerType = 'NSBH'

    elif Type == 'NSBH_LRR':
        ns_astro_mass_dist = stats.norm(1.33, 0.09)
        bh_astro_mass_dist = stats.pareto(b=1.3)
        m1 = bh_astro_mass_dist.rvs(size=mass_draws)
        m2 = ns_astro_mass_dist.rvs(size=mass_draws)
        MergerType = 'NSBH'

    elif Type == 'BNS_LRR':
        ns_astro_mass_dist = stats.norm(1.33, 0.09)
        m1 = ns_astro_mass_dist.rvs(size=mass_draws)
        m2 = ns_astro_mass_dist.rvs(size=mass_draws)
        MergerType = 'BNS'

    # sort to make sure m1 > m2
    for i, (m_a, m_b) in enumerate(zip(m1, m2)):
        if m_a < m_b:
            m1[i], m2[i] = m_b, m_a

    return m1, m2, MergerType


def run_EOS(EOS, m1, m2, thetas, N_EOS = 100, eospostdat = None, EOS_draws = None, EOS_idx = None):
    '''Pick EOS and calculate ejecta quantities

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
       eospostdat: str
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
    model_set = 'Bu2019inc'
    N_masses = len(m1) 
    
    q = m2/m1
    mchirp = (m1*m2)**(3/5) / (m1+m2)**(1/5)
    eta = m1*m2/((m1+m2)*(m1+m2))
  
    #fix spin lanthanide fraction value 
    chi, Xlan = 0, 1e-3  
   
    data = np.vstack((m1,m2,chi,mchirp,eta,q)).T
    samples = KNTable((data), names = ('m1','m2','chi_eff','mchirp','eta','q'))
    samples = EOS_samples(samples, thetas, N_EOS, eospostdat, EOS_draws, EOS_idx)
    
    print("m1: %.5f +-%.5f"%(np.mean(samples["m1"]),np.std(samples["m1"])))
    print("m2: %.5f +-%.5f"%(np.mean(samples["m2"]),np.std(samples["m2"])))
       
    
    # Downsample removed for now
    #samples = samples.downsample(Nsamples=100)

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
 
        from gwemlightcurves.EjectaFits.PaDi2019 import calc_meje, calc_vej
        # calc the mass of ejecta
        mej1, dyn_mej1, wind_mej1 = calc_meje(samples['m1'], samples['c1'], samples['m2'], samples['c2'], split_mej=True)
        # calc the velocity of ejecta
        vej1 = calc_vej(samples['m1'],samples['c1'],samples['m2'],samples['c2'])
        samples['mchirp'], samples['eta'], samples['q'] = lightcurve_utils.ms2mc(samples['m1'], samples['m2'])
    
        from gwemlightcurves.EjectaFits.KrFo2019 import calc_meje, calc_vave
        # calc the mass of ejecta
        mej2, dyn_mej2, wind_mej2 = calc_meje(samples['q'],samples['chi_eff'],samples['c2'], samples['m2'], split_mej=True)
        # calc the velocity of ejecta
        vej2 = calc_vave(samples['q'])
 
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
        erroropt = 'none'
        if erroropt == 'none':
            print("Not applying an error to mass ejecta")
        elif erroropt == 'log':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.236))
        elif erroropt == 'lin':
            samples['mej'] = np.random.normal(samples['mej'],0.72*samples['mej'])
        elif erroropt == 'loggauss':
            samples['mej'] = np.power(10.,np.random.normal(np.log10(samples['mej']),0.312))
    
        idx = np.where(samples['mej'] <= 0)[0]
        samples['mej'][idx] = 1e-11
            
        if (model_set == "Bu2019inc"):  
                idx = np.where(samples['mej'] <= 1e-6)[0]
                samples['mej'][idx] = 1e-11
        elif (model_set == "Ka2017"):
                idx = np.where(samples['mej'] <= 1e-3)[0]
                samples['mej'][idx] = 1e-11
 
        print("Probability of having ejecta")
        print(100 * (len(samples) - len(idx)) /len(samples))
        return samples


def EOS_samples(samples, thetas, nsamples, eospostdat, EOS_draws, EOS_idx, EOS = 'gp'):
    '''Draws different EOS's for ejecta calculations

       Parameters
       ----------
       samples: KNTable object
           table of ejecta quantities
       thetas: np.array
           array of theta draws
       nsamples: int
           number of EOS draws
       eospostdat: str
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

    #set spin to 0
    chi, Xlan = 0, 1e-3

    lambda1s, lambda2s, m1s, m2s = [], [], [], []
    chi_effs, Xlans, qs, mbnss = [], [], [], []

    m1s, m2s, dists_mbta = [], [], []
    lambda1s, lambda2s, chi_effs, mbnss = [], [], [], []   
 
    # read Phil + Reed's EOS files
    idxs = np.array(eospostdat["eos"])
    weights = np.array([np.exp(weight) for weight in eospostdat["logweight_total"]])

    Xlan_min, Xlan_max = -9, -1 
 
    for ii, row in enumerate(samples): 
        # m1, m2, dist_mbta, chi_eff = row["m1"], row["m2"], row["dist_mbta"], row["chi_eff"]
        m1, m2, chi_eff = row["m1"], row["m2"], row["chi_eff"]
        #elif EOS == "gp":
        # Note: fix weights
        indices = np.random.choice(np.array(EOS_idx), size=nsamples, replace=True)
        for jj in range(nsamples):
            if EOS == "gp":
                #index = gp10_idx[jj]
                index = indices[jj] 
                lambda1, lambda2 = -1, -1
                mbns = -1
                # samples lambda's from Phil + Reed's files
                #elif EOS == "gp":
                while (lambda1 < 0.) or (lambda2 < 0.) or (mbns < 0.):
                    phasetr = 0
                    data_out = EOS_draws[index] 
                    marray, larray = data_out["M"], data_out["Lambda"]
                    f = interp.interp1d(marray, larray, fill_value=0, bounds_error=False)
                    if float(f(m1)) > lambda1: lambda1 = f(m1) # pick lambda from least compact stable branch
                    if float(f(m2)) > lambda2: lambda2 = f(m2)
                    if np.max(marray) > mbns: mbns = np.max(marray) # get global maximum mass

            m1s.append(m1)
            m2s.append(m2)
            lambda1s.append(lambda1)
            lambda2s.append(lambda2)
            chi_effs.append(chi)
            Xlans.append(Xlan)
            mbnss.append(mbns)

    idx_thetas = np.where(thetas > 90.)[0]
    thetas[idx_thetas] = 180. - thetas[idx_thetas]
    Xlans = np.ones(np.array(m1s).shape) * Xlan

    # make final arrays of masses, distances, lambdas, spins, and lanthanide fractions
    data = np.vstack((m1s,m2s,lambda1s,lambda2s,Xlans,chi_effs,thetas,mbnss)).T
    samples = KNTable(data, names=('m1', 'm2', 'lambda1', 'lambda2','Xlan','chi_eff','theta', 'mbns'))

    return samples 
