import os
import numpy as np
import pandas as pd
from configparser import ConfigParser
from pathlib import Path
from calc_lightcurves import lightcurve_predictions
from mass_distributions import BNS_alsing, BNS_farrow, NSBH_zhu
from lightcurve_utils import load_eos_posterior
import pickle 
import h5py
import sqlite3

# load configs
rel_path = 'etc/conf.ini'
conf_path = Path(__file__).parents[3] / rel_path
config = ConfigParser()
config.read(conf_path)

# load ejecta configs
ejecta_model = eval(config.get('lightcurve_configs', 'ejecta_model'))
eos_config = ejecta_model['eosname']

#PE_path = '/home/andrew.toivonen/mdc-analytics/PE_samples/event_files_new'
#PE_path = '/home/andrew.toivonen/mdc-analytics/PE_updated/1350240000_1354278000'
PE_path = f'/home/andrew.toivonen/mdc-analytics/PE_updated/MDC9_PE/1353696000_1357152000/'
dirs = os.listdir(PE_path)

filename = "/home/andrew.toivonen/mdc-analytics/injections-minSNR-4.hdf5"
datafile = h5py.File(filename, "r")
key = list(datafile.keys())[0]
inj_df = pd.read_hdf(filename, key)
t_inj = inj_df['time_H']
m1_inj, m2_inj = inj_df['mass1'], inj_df['mass2']
distance, inclination = inj_df['distance'], inj_df['inclination']
# rad to deg
inclination = 180/np.pi*inclination
# ---------------------------------------------
# NOTE: SLy needs to be set in confi.ini!!!!!!!
# ---------------------------------------------
N_inj = 1
N_eos = 1
N_cores = 1


def run_lc(m1, m2, theta, d, event):
        try:
            lc_data, ejecta_data, yields_ejecta, eos_metadata, lightcurve_metadata = lightcurve_predictions(m1s = [m1], m2s = [m2], thetas = np.array([theta]), distances = d, N_eos=N_eos, N_cores=N_cores)
        except ValueError: return
        draws = int(len(ejecta_data)/N_eos)
        N_downsample = len(lc_data)
        #path = f'./PE_mdc/inj_lcs/{event}'
        #path = f'./PE_mdc/Sly_injs_later_mdcs/{event}'
        #path = f'./mdc_analysis/APR4_EPP_injs_updated_eos/{event}'
        #path = f'./mdc_analysis/H4_injs_updated_eos/{event}'
        #path = f'./mdc_analysis/SLy_MDC9/{event}'
        #path = f'./mdc_analysis/APR4_EPP_MDC9/{event}'
        #----------------------------------------------
        #path = f'./mdc_analysis/updated_fits/SLy_MDC9/{event}'
        path = f'./mdc_analysis/updated_fits2/APR4_EPP_MDC9/{event}'
        #path = f'./mdc_analysis/updated_fits2/SLy_MDC9/{event}'
        if not os.path.isdir(path):
            os.makedirs(path)
        filename = f'{path}/inj_lc_{N_downsample}_table_{event}_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(lc_data, f)
        filename = f'{path}/inj_ejecta_table_{event}_{eos_config}_{draws}x{N_eos}_{yields_ejecta}.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(ejecta_data, f)

def shift_mdc_time(t):
    '''use offsets to shift mdc times
    '''
    # offsets for gracedb_set_1326048000_1339872000
    #offsets = [87936000, 91392000]
    # MDC9
    offsets = [91392000]
    start_time = 1353696000
    end_time = 1357152000
    # times are inclusive, exclusive ie: [..)
    #offset_times = [[1350240000, 1353696000], [1353696000, 1357152000]]
    offset_times = [[1353696000, 1357152000]]
    shifted_times = []
    for i, t_range in enumerate(offset_times):
        if (t_range[0] <= float(t) < t_range[1]):
            shifted_times.append(float(t) - offsets[i])
            #j=i+8
            j = 9
            print(f'Found in MDC{j}')
        else: 
            print(f'Not within MDC{j}')
    return(shifted_times)

for event in dirs:
    print(event)
    try:
        path = f'{PE_path}/{event}/Bilby.posterior_samples.hdf5'
        df = pd.DataFrame(np.array(h5py.File(path)['posterior_samples']))
        t = df['time'].values
        #print(t, t_inj)
        t_shift = shift_mdc_time(t[0])
        if t_shift:
            idx = np.where(np.abs(t_inj-t_shift) < 1)[0]
            if len(idx) == 1: # was > 0
                m1, m2 = m1_inj[idx[0]], m2_inj[idx[0]]
                d, theta = distance[idx[0]], inclination[idx[0]] 
                run_lc(m1, m2, theta, d, event)
            else:
                print('Event outside of our MDC range')   
 
    except FileNotFoundError:
        print(f'No posterior samples found for {event}')
        continue
