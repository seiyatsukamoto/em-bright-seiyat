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

# load configs
rel_path = 'etc/conf.ini'
conf_path = Path(__file__).parents[3] / rel_path
config = ConfigParser()
config.read(conf_path)

# load ejecta configs
ejecta_model = eval(config.get('lightcurve_configs', 'ejecta_model'))
eos_config = ejecta_model['eosname']

PE_path = '/home/andrew.toivonen/mdc-analytics/PE_samples/event_files_new'
dirs = os.listdir(PE_path)

filename = '/home/andrew.toivonen/farah/O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_all.h5'
RP_events = Table.read(filename)
m1, m2 = RP_events['mass_1'], RP_events['mass_2']
RP_events['mchirp'] = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))

NSmin = 1.2768371655
#NSmax = 1.4402725191

RP_events = RP_events[RP_events['mass_1'] >= NSmin]
RP_events = RP_events[RP_events['mass_2'] >= NSmin]

downsample_PE = False
#downsample_PE = True
N_inj = 20
N_eos = 30
#N_inj = 1
#N_eos = 1
N_cores = 20
#N_cores = 1

#theta = [45]
#event_name = 'S220912bi'
#path =  '/home/andrew.toivonen/mdc-analytics/PE_samples/event_files/Bilby.posterior_samples.hdf5'

def run_lc(m1, m2, d, event):
        start = time.time()
        #lc_data, yields_ejecta, eos_metadata, lightcurve_metadata = lightcurve_predictions(m1s = m1, m2s = m2, distances = d, N_eos=N_eos, N_cores=N_cores)
        lc_data, ejecta_data, yields_ejecta, eos_metadata, lightcurve_metadata = lightcurve_predictions(m1s = m1, m2s = m2, N_eos=N_eos, N_cores=N_cores)
        stop = time.time()
        t = round((stop-start), 1)
        print(f'Time of run: {t}s')
        draws = int(len(ejecta_data)/N_eos)
        N_downsample = len(lc_data)
        #path = f'./PE_mdc/PE_lcs/{event}'
        #path = f'./PE_mdc/PE_with_RP/{event}'
        path = f'./PE_mdc/PE_with_RP/{event}'
        if not os.path.isdir(path):
            os.makedirs(path)
        yields_ejecta = round(yields_ejecta, 3)
        filename = f'{path}/PE_lc_{N_downsample}_table_{event}_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(lc_data, f)
        filename = f'{path}/PE_ejecta_table_{event}_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(ejecta_data, f)

#dirs = ['S220903gx']
#dirs = ['S220916fk']
for event in dirs:
    print(event)
    try:
        path = f'{PE_path}/{event}/Bilby.posterior_samples.hdf5'
        PE_data = pd.DataFrame(np.array(h5py.File(path)['posterior_samples']))
       
        distances = PE_data['luminosity_distance']
        shift_distances = True
        if shift_distances:
            print('Shifting masses using passed distances beforehand')
            distances = Distance(distances, u.Mpc)
            z = distances.compute_z(Planck18)
            #m1s = m1s/(1+z)
            #m2s = m2s/(1+z)
            PE_data['mass_1'] = PE_data['mass_1']/(1+z)
            PE_data['mass_2'] = PE_data['mass_2']/(1+z)

        if downsample_PE:
            rand_idx = np.random.choice(len(PE_data), size = N_inj)
            PE_data = PE_data.loc[rand_idx]
        
        m1, m2 = PE_data['mass_1'].values, PE_data['mass_2'].values
        d = PE_data['luminosity_distance'].values
        
        m1_unsorted, m2_unsorted, d = np.array(m1), np.array(m2), np.array(d)

        #wrong_m1 = m1_unsorted[m2_unsorted > m1_unsorted]
        #wrong_m2 = m2_unsorted[m2_unsorted > m1_unsorted]

        
        m1 = np.maximum(m1_unsorted, m2_unsorted)
        m2 = np.minimum(m1_unsorted, m2_unsorted) 

        PE_data['mchirp'] = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))

        print(len(PE_data))
        #if len(wrong_m1) > 0:

        # source/detector frame?
        min_mc, max_mc = np.min(PE_data['mchirp']), np.max(PE_data['mchirp'])
        #print(min_mc, max_mc)

        # find events within mchirp range
        # Widen the range +/- 10%?????
        #RP_events_cut = RP_events[RP_events['mchirp'] >= min_mc]
        #RP_events_cut = RP_events_cut[RP_events_cut['mchirp'] <= max_mc]
        RP_events_cut = RP_events[RP_events['mchirp'] >= .9*min_mc]
        RP_events_cut = RP_events_cut[RP_events_cut['mchirp'] <= 1.1*max_mc]

        print(len(RP_events_cut))
        if len(RP_events_cut) > 0:
            q_draws = np.random.choice(RP_events_cut['mass_ratio'], len(PE_data))
            eta_draws = lightcurve_utils.q2eta(q_draws)
            m1_draws, m2_draws = lightcurve_utils.mc2ms(PE_data['mchirp'].values, eta_draws)
            print(m1_draws, m2_draws)
            #if len(wrong_m1) > 0:
            run_lc(m1_draws, m2_draws, d, event)

    except FileNotFoundError:
        print(f'No posterior samples found for {event}')
        continue
