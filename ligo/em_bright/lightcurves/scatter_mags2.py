import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_with_RP'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_lcs'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_all_posterior'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_lcs/Sly'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/inj_lcs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/SLy_injs'

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/output/grid'
events = os.listdir(event_path)

#band_plot = 'r'

def m1_vs_m2(filepath, lc_filename):
    '''Function to plot heat map of lightcurves
    '''
    path = f'{filepath}/{lc_filename}'
    with open (path, 'rb') as f:
        data = pickle.load(f)
    # all ejecta file (due to downsampling)
    '''
    ej_path = f'{filepath}/{ej_filename}'
    with open (ej_path, 'rb') as f:
        ej_data = pickle.load(f)
    '''
    '''
    try:
        inj = f'{inj_path}/{event}/inj_lc_table_{event}_SLy_1x1_1.0.pickle'
        with open (inj, 'rb') as f:
            inj_data = pickle.load(f)
    except FileNotFoundError:
        inj = f'{inj_path}/{event}/inj_lc_table_{event}_SLy_1x1_0.0.pickle'
        with open (inj, 'rb') as f:
            inj_data = pickle.load(f)
    '''
    lc_dict, lc_dict_inj = {}, {}
    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']
    #bands = ['r']

    for band in bands:
        lc_dict[band], lc_dict_inj[band] = [], []

    for line in data:
        t = line['t']
        mags = line['mag']
        for idx, mag in enumerate(mags):
            if bands[idx] == band_plot:
                lc_dict[bands[idx]].append(mag)
    '''        
    for line in inj_data:
        #t = line['t']
        inj_mags = line['mag']
        for idx, mag in enumerate(inj_mags):
            if bands[idx] == band_plot:
                lc_dict_inj[bands[idx]].append(mag)
    '''
    # fine if NO DOWNSAMPLING
    ej_data = data

    mej_data = data['mej']
    theta_data = data['theta']
    merger_type = data['merger_type']

    all_mej_data = ej_data['mej']
    all_theta_data = ej_data['theta']
    all_merger_type = ej_data['merger_type']

    # only calculate lightcurves for mej > 1e-3
    sig_ej_idx = np.where(mej_data>1e-3)[0]
    all_sig_ej_idx = np.where(all_mej_data>1e-3)[0]
    has_ejecta = len(all_sig_ej_idx)/len(ej_data)
    #print(f'Significant ejecta: {has_ejecta}')

    mej_data = mej_data[sig_ej_idx]
    theta_data = theta_data[sig_ej_idx]
    merger_type = merger_type[sig_ej_idx]
    all_merger_type = all_merger_type[all_sig_ej_idx]

    # BNS_ejecta, NSBH_ejecta
    lc_count = len(ej_data)
    BNS_count = len(np.where(all_merger_type == 'BNS')[0])
    NSBH_count = len(np.where(all_merger_type == 'NSBH')[0])
    BNS_frac = BNS_count/lc_count
    NSBH_frac = NSBH_count/lc_count


    if True:
        band = band_plot
        lc  = np.array(lc_dict[band])
        lc = lc[sig_ej_idx]


        percentiles = {}
        percentile_list = [5, 50, 95]
        peak_mags = {}
        percentiles[band] = np.nanpercentile(lc, percentile_list, axis=0)

        percentiles['mej'] = np.nanpercentile(np.array(mej_data), percentile_list)
        percentiles['theta'] = np.nanpercentile(np.array(theta_data), percentile_list)

        # mag opposite order due to nature of magnitudes
        mej5, mej50, mej95 = percentiles['mej'][0], percentiles['mej'][1], percentiles['mej'][2]
        mag5, mag50, mag95 = percentiles[band][2], percentiles[band][1], percentiles[band][0]

        peak_5, peak_50, peak_95 = np.min(mag5), np.min(mag50), np.min(mag95)

        return peak_5, peak_50, peak_95

m1s = np.linspace(1,2.1,20)
m2s = np.linspace(1,2.1,20)
m1_grid, m2_grid = np.meshgrid(m1s, m2s)
m1_plot, m2_plot = [], []
for m1, m2 in zip(m1_grid.flatten(), m2_grid.flatten()):
    if m1 > m2: 
        m1_plot.append(m1)
        m2_plot.append(m2)

#if __name__ == "__main__":
bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']
bands = ['r']
for band_plot in bands:
    m1s, m2s, peaks5, peaks50, peaks95 = [], [], [], [], []
    #for event in events:
    for m1, m2 in zip(m1_plot, m2_plot):
        print(m1, m2)
        try:
            #lc_prefix = 'PE_lc'
            #ej_prefix = 'PE_ejecta'
            lc_prefix = f'lc_table_{m1}_{m2}x60'
            filepath = event_path
            files = os.listdir(filepath)
            #print(files)
            lc_files = [f for f in files if f.startswith(lc_prefix)]
            print(lc_files)
        except FileNotFoundError:
            continue
        for lc_filename in lc_files:
            #m1_vs_m2(filepath, lc_filename, ej_filename, event)
            try:
                peak5, peak50, peak95 = m1_vs_m2(filepath, lc_filename)
                m1s.append(m1)
                m2s.append(m2)
                peaks5.append(peak5)
                peaks50.append(peak50)
                peaks95.append(peak95)
            except:
                plt.close('all')
                continue

    idxs=np.where(np.array(m1s) <= 1.95)[0]
    m1s, m2s = np.array(m1s)[idxs], np.array(m2s)[idxs]
    peaks5, peaks50, peaks95 = np.array(peaks5)[idxs], np.array(peaks50)[idxs], np.array(peaks95)[idxs]
    idxs=np.where(np.array(m2s) >= 1.1)[0]
    m1s, m2s = np.array(m1s)[idxs], np.array(m2s)[idxs]
    peaks5, peaks50, peaks95 = np.array(peaks5)[idxs], np.array(peaks50)[idxs], np.array(peaks95)[idxs]

    #f,axes=plt.subplots(ncols=3, figsize=(12,9), sharey=True, sharex=True)
    #im0 = axes[0].scatter(m1s, m2s, c=peaks5, cmap='viridis_r')
    #im1 = axes[1].scatter(m1s, m2s, c=peaks50, cmap='viridis_r')
    #im2 = axes[2].scatter(m1s, m2s, c=peaks95, cmap='viridis_r')
    #axes[0].set_ylabel(r'm2 $[M_\odot]$')
    #axes[1].set_xlabel(r'm1 $[M_\odot]$')
    #plt.xlim([1,2.2])

    #cbar0=f.colorbar(im0, label=r'$M_{AB}$'+f' peak of 5th percentile {band_plot} band lightcurve', ax=axes[0], aspect=40)
    #cbar1=f.colorbar(im1, label=r'$M_{AB}$'+f' peak of 50th percentile {band_plot} band lightcurve', ax=axes[1], aspect=40)
    #cbar2=f.colorbar(im2, label=r'$M_{AB}$'+f' peak of 95th percentile {band_plot} band lightcurve', ax=axes[2], aspect=40)
    #cbar0=f.colorbar(im0, label=r'$M_{AB}$'+f' peak of 5th percentile {band_plot} band lightcurve', ax=axes[0], location='top')
    #cbar1=f.colorbar(im1, label=r'$M_{AB}$'+f' peak of 50th percentile {band_plot} band lightcurve', ax=axes[1], location='top')
    #cbar2=f.colorbar(im2, label=r'$M_{AB}$'+f' peak of 95th percentile {band_plot} band lightcurve', ax=axes[2], location='top')
    #cbar0=f.colorbar(im0, label=f'peak {band_plot}'+r' $M_{AB}$'+ ' for 5th percentile lightcurve', ax=axes[0], location='top')
    #cbar1=f.colorbar(im1, label=f'peak {band_plot}'+r' $M_{AB}$'+ ' for 50th percentile lightcurve', ax=axes[1], location='top')
    #cbar2=f.colorbar(im2, label=f'peak {band_plot}'+r' $M_{AB}$'+ ' for 95th percentile lightcurve', ax=axes[2], location='top')
    #cbar0.ax.invert_xaxis()
    #cbar1.ax.invert_xaxis()
    #cbar2.ax.invert_xaxis()

    plt.figure(figsize=(12,9))
    plt.scatter(m1s, m2s, c=peaks50, s=200, cmap='viridis_r')
    plt.xlabel(r'm1 $[M_\odot]$')
    plt.ylabel(r'm2 $[M_\odot]$')
    #plt.xlim([1,4])
    plot_filename = f'./output/scatter_mags_{band_plot}.png'
    print(f'Saving: {plot_filename}')
    cbar=plt.colorbar(label=r'$M_{AB}$'+f' peak of 50th percentile {band_plot} band lightcurve')
    cbar.ax.invert_yaxis()
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close('all')
