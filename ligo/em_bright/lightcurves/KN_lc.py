import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from pathlib import Path
from calc_lightcurves import lightcurve_predictions
from mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu
from lightcurve_utils import load_eos_posterior
from gwemlightcurves import lightcurve_utils
import time
import pickle 
import h5py
from astropy.table import Table
from astropy.cosmology import Planck18
from astropy.coordinates import Distance
from astropy import units as u

import pdb
#-----------------------------------------------
event = 'S230518h'
#-----------------------------------------------


# load configs
rel_path = 'etc/conf.ini'
conf_path = Path(__file__).parents[3] / rel_path
config = ConfigParser()
config.read(conf_path)

# load ejecta configs
ejecta_model = eval(config.get('lightcurve_configs', 'ejecta_model'))
eos_config = ejecta_model['eosname']

PE_path = f'./O4/{event}/PE'
#dirs = os.listdir(PE_path)

N_eos = 50
#N_eos = 1
#N_cores = 10
#N_cores = 3
N_cores = 1

def run_lc(mass_1, mass_2, luminosity_distance):
        path = f'./O4/{event}'
        lc_data, ejecta_data, yields_ejecta, eos_metadata, lightcurve_metadata = lightcurve_predictions(m1s = mass_1, m2s = mass_2, distances=luminosity_distance, N_eos=N_eos, N_cores=N_cores)
        draws = int(len(ejecta_data)/N_eos)
        path = f'./O4/{event}'
        if not os.path.isdir(path):
            os.makedirs(path)
        yields_ejecta = round(yields_ejecta, 3)
        try:
            #filename = f'{path}/PE_lc_{len(lc_data)}_table_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
            filename = f'{path}/PE_lc_Bu2019lm_{len(lc_data)}_table_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
            with open(filename, 'wb') as f:
                pickle.dump(lc_data, f)
        except: pass
        filename = f'{path}/PE_ejecta_table_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(ejecta_data, f)


def m1_greater_than_m2(PE_data):
    m1_unsorted, m2_unsorted = np.array(PE_data['mass_1']), np.array(PE_data['mass_2'])

    PE_data['mass_1'] = np.maximum(m1_unsorted, m2_unsorted)
    PE_data['mass_2'] = np.minimum(m1_unsorted, m2_unsorted)
    return PE_data


def load_bilby_PE(filename):
    PE_data = pd.DataFrame(np.array(h5py.File(filename)['posterior_samples']))
    if not ('mass_1_source' and 'mass_2_source') in PE_data.keys():
        distances = Distance(PE_data['luminosity_distance'].values, u.Mpc)
        z = distances.compute_z(Planck18)
        PE_data['mass_1_source'] = PE_data['mass_1']/(1+z)
        PE_data['mass_2_source'] = PE_data['mass_2']/(1+z)
    return(PE_data)

def downsample_PE(PE_data, N_downsample=100):
    rand_idx = np.random.choice(len(PE_data), size = N_downsample)
    return PE_data.loc[rand_idx]

PE_path = f'./O4/{event}/PE/Bilby.posterior_samples.hdf5'
PE_data = load_bilby_PE(PE_path)
print(PE_data.keys())
downsample = True
if downsample:
    PE_data = downsample_PE(PE_data)

m1, m2 = PE_data['mass_1_source'].values, PE_data['mass_2_source'].values
d = PE_data['luminosity_distance'].values

run_lc(m1, m2)
