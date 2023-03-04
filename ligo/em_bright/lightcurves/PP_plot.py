import os
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import pickle5 as pickle

eos = 'SLy'
#eos = 'APR4_EPP'
#eos = 'H4'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_with_RP'
#  event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_later_MDC'
#  inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/SLy_injs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/Sly_injs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/SLy_injs_updated_eos'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_MDC9_1000'
#inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/{eos}_MDC9'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_q_MDC9_1000'
#inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/{eos}_MDC9'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits/PE_MDC9_200'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits/PE_MDC9_100'
#inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits/{eos}_MDC9'

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/PE_MDC9_500'
inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/{eos}_MDC9'

events = os.listdir(inj_path)
#events = events[:50]


def PP_plot(percentiles, quantity, band):
    '''
    '''
    
    #print(percentiles)
    percentiles.sort()
    #print(percentiles)
    #cdf = np.cumsum(percentiles)
    cdf = np.cumsum(np.ones(len(percentiles)))
    cdf = cdf/np.max(cdf)
    #bins = percentiles


    '''
    percentiles = np.abs(np.array(percentiles)-50)*2
    percentiles.sort()

    N_bins = len(percentiles)
    print(f'{N_bins} events plotted')
    hist, bins = np.histogram(percentiles, bins = N_bins)
    cdf = np.cumsum(hist)
    cdf = cdf/np.max(cdf)
    '''   

    '''
    plt.figure(figsize=(8,6)) 
    plt.hist(percentiles, bins=20)
    plt.xlabel(f'{quantity} Credible Interval: abs(Percentile-50)*2')
    plt.ylabel('Probability')
    plt.savefig(f'paper_plots/PP_plots/hist_RP_per_from_50_{quantity}.png')
    plt.close()
    '''

    #plt.figure(figsize=(12,9))
    #plt.hist(data['q'], bins=20, alpha = .8, edgecolor = 'black', linewidth = 2, density = 1)
    #cdf[-1] = cdf[-2]
    #print(event)
    percentiles=np.array(percentiles)
    cred80 = len(percentiles[percentiles <= 80])/len(percentiles)   
    cred90 = len(percentiles[percentiles <= 90])/len(percentiles)

    #print(f'Fraction of Events in the 80% Credible Interval: {cred80} ({band})')
    #print(f'Fraction of Events in the 90% Credible Interval: {cred90} ({band})')    
    if quantity != 'mag':
        percentiles.flatten()
        plt.plot(percentiles/100, cdf, linewidth=5, label = f'{quantity}')
    if quantity == 'mag':
        plt.plot(percentiles/100, cdf, linewidth=5, label = f'{band} band')
    #cut_percentiles, cut_cdf = percentiles[percentiles < 100], cdf[percentiles < 100]
    #cdf[percentiles == 100.] = cut_cdf[-1]
    #plt.plot([np.min(percentiles), np.max(percentiles)], [0, 1], color='black')
    #plt.axvline(90, color='k', linestyle='dashed', label = '90 Percent Credible Interval')
    plt.ylim([0,1])
    plt.xlim([0,1])
    plt.plot([0,1], [0, 1], color='black')
    #plt.xlabel('Credible Interval')
    plt.xlabel('Percentile')
    if quantity == 'mej':
        if not multiple:
            #plt.ylabel(r'Fraction of SLy $m_{ej}$ values within Credible Interval of Predictions')
            plt.ylabel(f'Fraction of {eos} '+r' $m_{ej}$ values')
    if quantity == 'mag':
        bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']
        band_idx = 0
        #plt.ylabel(r'Fraction of SLy $M_{AB}$ values within Credible Interval of Predictions')
        plt.ylabel(f'Fraction of {eos} '+r'$M_{AB}$ values')
    else:
        if not multiple:
            #plt.ylabel(f'Fraction of {eos} {quantity} values within Credible Interval of Predictions')
            plt.ylabel(f'Fraction of {quantity} values')
    if multiple:
        plt.ylabel(f'Fraction of values')
    #plt.legend()
    #plt.savefig(f'paper_plots/PP_plots/PP_plot_RP_{quantity}.png')
    if quantity != 'mag':
        if not multiple: 
            #plt.savefig(f'paper_plots/PP_plots/PP_plot_RP_{quantity}.png')
            #plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_MDC9_{eos}_updated_{quantity}.png')
            # plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_old_{eos}_updated_{quantity}.png')
            #plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_q_MDC9_{eos}_{quantity}.png')
            plt.savefig(f'mdc_analysis/plots/updated_fits2/PP_plots/PP_plot_MDC9_{eos}_{quantity}.png')
            plt.close()
    if quantity == 'mag':
        if band == 'K':
            #plt.axvline(.90, color='k', linestyle='dashed', label = '90% Credible Interval')
            plt.legend()
            #plt.savefig(f'paper_plots/PP_plots/PP_plot_RP_{quantity}.png')
            #plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_MDC9_{eos}_{quantity}.png')
            #plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_old_{eos}_{quantity}.png')
            #plt.savefig(f'mdc_analysis/plots/PP_plots/PP_plot_q_MDC9_{eos}_{quantity}.png')
            plt.savefig(f'mdc_analysis/plots/updated_fits2/PP_plots/PP_plot_MDC9_{eos}_{quantity}.png')
            plt.close()

def PP_percentiles(data, inj):
    '''
    '''
    per = percentileofscore(data, inj)

#if __name__ == "__main__":
def per_for_quantity(quantity, band, multiple=False):
    percentiles = []
    print(f'running for {quantity}, {band}')
    if quantity == 'mag':
        prefix = 'PE_lc'
    else:
        prefix = 'PE_ejecta'
    #events = ['S221021bm'] 
    for event in events:
        print(event)
        try:
            filepath = f'{event_path}/{event}'
            files = os.listdir(filepath)
            files = [f for f in files if f.startswith(prefix)]
        except FileNotFoundError:
            continue
        for filename in files:
            try:
                path = f'{filepath}/{filename}'
                with open (path, 'rb') as f:
                    data = pickle.load(f)
                for line in data:
                    if line['m2'] > line['m1']:
                        print(f'q > 1: {event}')
                # use all files, even those with small mej
                '''
                if quantity != 'mag':
                    inj_prefix = 'inj_lc'
                    inj_filepath = f'{inj_path}/{event}'
                    inj_files = os.listdir(inj_filepath)
                    inj_files = [f for f in inj_files if f.startswith(inj_prefix)]
                    inj_f = inj_files[0]
                    inj = f'{inj_path}/{event}/{inj_f}'
                '''
                #else:
                if True:
                    inj = f'{inj_path}/{event}/inj_lc_1_table_{event}_{eos}_1x1_1.0.pickle'
                with open (inj, 'rb') as f:
                    inj_data = pickle.load(f)
                if quantity == 'mag':
                    lc_dict, lc_dict_inj = {}, {}
                    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

                    for band_dict in bands:
                        lc_dict[band_dict], lc_dict_inj[band_dict] = [], []
                    #lc_dict[band], lc_dict_inj[band] = [], []

                    for line in data:
                        t = line['t']
                        mags = line['mag']
                        for idx, mag in enumerate(mags):
                            lc_dict[bands[idx]].append(mag)
                            #lc_dict[band].append(mag)

                    for line in inj_data:
                        #t = line['t']
                        inj_mags = line['mag']
                        for idx, mag in enumerate(inj_mags):
                            lc_dict_inj[bands[idx]].append(mag)
                            #lc_dict_inj[band].append(mag)

                    #band = 'r'
                    lc  = np.array(lc_dict[band])
                    lc_inj = np.array(lc_dict_inj[band])[0]
                    peak_mag_inj = np.min(lc_inj)
                    lc_mins = [np.min(line) for line in lc]
                    print(len(lc_mins))
                    #print(lc_mins)
                    if len(lc_mins) > 1:
                        percentiles.append(percentileofscore(lc_mins, peak_mag_inj))
                    else: continue
                    ''' 
                    for idx, band in enumerate(bands):
                        lc  = np.array(lc_dict[band])
                        lc_inj = np.array(lc_dict_inj[band])[0]
                        peak_mag_inj = np.min(lc_inj)
                        percentiles.append(percentileofscore(np.minimum(lc), peak_mag_inj))
                    '''
                else:
                    #print((data[quantity]).dtype())
                    #print(data[quantity], inj_data[quantity])
                    #print(list(data[quantity]))
                    if len(data[quantity][:]) > 1:
                        percentiles.append(percentileofscore(data[quantity][:], inj_data[quantity])[0])
                    else: continue
            except FileNotFoundError:
                continue

    plt.figure(figsize=(8,6))
    plt.hist(percentiles, bins=20)
    plt.xlabel(f'{quantity} Percentile')
    plt.ylabel('Probability')
    plt.savefig(f'mdc_analysis/plots/updated_fits2/PP_plots/hist_{quantity}.png')
    plt.close()

    print('Number of events:')
    print(len(percentiles))
    #print(percentiles)
    
    if quantity == 'mag':
        percentiles = list(100-np.array(percentiles))
        plt.figure(figsize=(8,6))
        plt.hist(percentiles, bins=20)
        plt.xlabel(f'{quantity} Percentile')
        plt.ylabel('Probability')
        plt.savefig(f'mdc_analysis/plots/updated_fits2/PP_plots/hist_reverse_{quantity}.png')
        plt.close()
    
    return percentiles   


multiple = True
multiple = False
quantities = ['mag']
quantities = ['mej']

#quantities = ['mchirp']
#quantities = ['m1']
#quantities = ['q']
#quantities = ['theta']

#quantities = ['lambda1', 'lambda2']
#quantities = ['q', 'mchirp']

plt.figure(figsize=(12,9))
for quantity in quantities:
    lc_dict, lc_dict_inj = {}, {}
    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K'] 
    if quantity != 'mag':
        bands = [None]
    for band in bands:
        percentile_list = per_for_quantity(quantity, band)
        PP_plot(percentile_list, quantity, band)
if multiple:
    plt.legend()
    plt.savefig(f'mdc_analysis/plots/updated_fits/PP_plots/PP_plot_MDC9_multiple.png')
    plt.close()
