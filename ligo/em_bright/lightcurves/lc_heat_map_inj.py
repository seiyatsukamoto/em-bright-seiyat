import os
import numpy as np
import matplotlib.pyplot as plt
import pickle5 as pickle

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_with_RP'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_lcs'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_all_posterior'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_lcs/Sly'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_later_MDC'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/inj_lcs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'

eos = 'SLy'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits/PE_MDC9_1000'
#inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits/{eos}_MDC9'

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/PE_MDC9_500'
inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/{eos}_MDC9'

events = os.listdir(inj_path)
#events = events[:50]

def heat_map(filepath, lc_filename, ej_filename, event):
    '''Function to plot heat map of lightcurves
    '''
    path = f'{filepath}/{lc_filename}'
    with open (path, 'rb') as f:
        data = pickle.load(f)
    # all ejecta file (due to downsampling)
    ej_path = f'{filepath}/{ej_filename}'
    with open (ej_path, 'rb') as f:
        ej_data = pickle.load(f)
    try:
        #if True: 
        inj = f'{inj_path}/{event}/inj_lc_1_table_{event}_SLy_1x1_1.0.pickle'
        with open (inj, 'rb') as f:
            inj_data = pickle.load(f)
    except FileNotFoundError:
        return
    lc_dict, lc_dict_inj = {}, {}
    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

    for band in bands:
        lc_dict[band], lc_dict_inj[band] = [], []

    for line in data:
        t = line['t']
        mags = line['mag']
        for idx, mag in enumerate(mags):
            lc_dict[bands[idx]].append(mag)

    for line in inj_data:
        #t = line['t']
        inj_mags = line['mag']
        for idx, mag in enumerate(inj_mags):
            lc_dict_inj[bands[idx]].append(mag)

    f,axes=plt.subplots(ncols=5,nrows=2,figsize=(35,15),sharey='row')
    plt.rcParams['figure.dpi'] = 200
    plt.rc('xtick',labelsize=30)
    plt.rc('ytick',labelsize=30)

    tsteps = 150

    mej_data = data['mej']
    theta_data = data['theta']
    merger_type = data['merger_type']

    all_mej_data = ej_data['mej']
    all_theta_data = ej_data['theta']
    all_merger_type = ej_data['merger_type']

    # THIS WAS MOVED TO CALC_LIGHTCURVES
    # only calculate lightcurves for mej > 1e-3
    #sig_ej_idx = np.where(mej_data>1e-3)[0]
    #all_sig_ej_idx = np.where(all_mej_data>1e-3)[0]
    #has_ejecta = len(all_sig_ej_idx)/len(ej_data)
    #print(f'Significant ejecta: {has_ejecta}')

    has_ejecta_data = ej_data[ej_data['mej'] > 1e-3]
    has_ejecta = len(has_ejecta_data)/len(ej_data)
    print(f'has_ejecta: {has_ejecta}')

    #mej_data = mej_data[sig_ej_idx]
    #theta_data = theta_data[sig_ej_idx]
    #merger_type = merger_type[sig_ej_idx]
    #all_merger_type = all_merger_type[all_sig_ej_idx]

    print(len(ej_data), len(data))
    # BNS_ejecta, NSBH_ejecta
    lc_count = len(ej_data)
    #BNS_count = len(np.where(all_merger_type == 'BNS')[0])
    #NSBH_count = len(np.where(all_merger_type == 'NSBH')[0])
    BNS_count = len(np.where(has_ejecta_data['merger_type'] == 'BNS')[0])
    NSBH_count = len(np.where(has_ejecta_data['merger_type'] == 'NSBH')[0])
    BNS_frac = BNS_count/lc_count
    NSBH_frac = NSBH_count/lc_count

    mej_inj = inj_data['mej'][0]
    theta_inj = inj_data['theta'][0]
    print(mej_inj, theta_inj)

    legend_band = 'r'
 
    for (i,j,band) in zip([0,0,0,0,0,1,1,1,1],[0,1,2,3,4,0,1,2,3],bands):

        lc  = np.array(lc_dict[band])
        lc_inj = np.array(lc_dict_inj[band])[0]
        #lc = lc[sig_ej_idx]

        peak_mag_inj = np.min(lc_inj)

        bins = np.linspace(-20, 1, 50)
        X, Y = np.meshgrid(t, bins[:-1])

        hist2d_1 = np.apply_along_axis(lambda a: np.histogram(a, bins=bins)[0], 0, lc)

        hist2d_1 = hist2d_1.astype('float')
        hist2d_1[hist2d_1 == 0] = np.nan
        im = axes[i][j].pcolormesh(X, Y, hist2d_1, cmap='viridis',alpha=0.7)

        percentiles = {}
        #percentile_list = [10, 50, 90]
        percentile_list = [5, 50, 95]
        peak_mags = {}
        if len(lc) > 1:
            percentiles[band] = np.nanpercentile(lc, percentile_list, axis=0)
        else: return

        percentiles['mej'] = np.nanpercentile(np.array(mej_data), percentile_list)
        percentiles['theta'] = np.nanpercentile(np.array(theta_data), percentile_list)

        # mag opposite order due to nature of magnitudes
        mej10, mej50, mej90 = percentiles['mej'][0], percentiles['mej'][1], percentiles['mej'][2]
        mag10, mag50, mag90 = percentiles[band][2], percentiles[band][1], percentiles[band][0]

        if band == 'r':
            r_peak_10, r_peak_50, r_peak_90 = np.min(mag10), np.min(mag50), np.min(mag90)
            r_peak_inj = peak_mag_inj
            print(r_peak_10, r_peak_50, r_peak_90, peak_mag_inj)

        # label Source Classification
        #axes[i][j].plot([], [], ' ', label=f'Source Classification: \n{round(BNS_frac,3)} BNS, {round(NSBH_frac,3)} NSBH')

        # fraction of events with mej > 1e-3
        axes[i][j].plot([], [], ' ', label=f'{round(BNS_frac,3)} BNS_ejecta, {round(NSBH_frac,3)} NSBH_ejecta')

        # plot 10th, 50th, and 90th percentiles
        axes[i][j].plot(t, mag90, linestyle = '--', linewidth = 4, color = 'salmon', label=f'{percentile_list[2]}th percentile values:\n'+r'$M_{ej}$:'+f'{round(mej90,3)}, Peak {legend_band} '+r'$M_{AB}$'+f': {round(np.min(mag90),2)}')
    
        axes[i][j].plot(t, mag50, linestyle = '--', linewidth = 4, color = 'lightsalmon', label=f'{percentile_list[1]}th percentile values:\n'+r'$M_{ej}$:'+f'{round(mej50,3)}, Peak {legend_band} '+r'$M_{AB}$'+f': {round(np.min(mag50),2)}')

        axes[i][j].plot(t, mag10, linestyle = '--', linewidth = 4, color = 'peachpuff', label=f'{percentile_list[0]}th percentile values:\n'+r'$M_{ej}$:'+f'{round(mej10,3)}, Peak {legend_band} '+r'$M_{AB}$'+f': {round(np.min(mag10),2)}')

        axes[i][j].plot(t, lc_inj, linestyle = '--', linewidth = 4, color = 'black', label=f'Injected Lightcurve (SLy): \n'+r'$M_{ej}$:'+f'{round(mej_inj,3)}, Peak {legend_band} '+r'$M_{AB}$'+f': {round(peak_mag_inj,2)}'+f'\n Inclination: {round(theta_inj, 3)}'+u'\N{DEGREE SIGN}')

        h1, l1 = axes[i][j].get_legend_handles_labels()
        #h2, l2 = axes[1][1].get_legend_handles_labels()
        #Make the legend
        if band == legend_band:
            legend = axes[-1][-1].legend(h1, l1,  bbox_to_anchor=(0,1,1.0,-0.15), loc=9,
                     ncol=1,prop={'size': 20},fancybox=True,frameon=True)

            frame = legend.get_frame()
            #frame.set_color('skyblue')
            frame.set_color('mediumaquamarine')

            #if band == 'K':
            cb_ax = f.add_axes([0.94, 0.14, 0.023, 0.7])
            cb = f.colorbar(im, cax = cb_ax, ticks=[], label = 'Less Events                                                       More Events')
            #cb.set_label('More Events', size=30)
        axes[i][j].set_ylim([0,-20])
        axes[i][j].text(10,-17,f'{band}',size=30)
        axes[i][j].set_xlim([0, 15])    
    f.text(0.5,0.05,'Time [days]',size=30)
    axes[0][0].set_ylabel('$M_{AB}$',size=30)
    axes[1][0].set_ylabel('$M_{AB}$',size=30)

    axes[-1, -1].axis('off')

    '''
    h1, l1 = axes[0][0].get_legend_handles_labels()
    #h2, l2 = axes[1][1].get_legend_handles_labels()

    #Make the legend
    legend = axes[-1][-1].legend(h1, l1,  bbox_to_anchor=(0,1,1.0,-0.15), loc=9,
             ncol=1,prop={'size': 20},fancybox=True,frameon=True)

    frame = legend.get_frame()
    #frame.set_color('skyblue')
    frame.set_color('mediumaquamarine')
    '''

    label = lc_filename.removeprefix('PE_lc_table_')
    label = label.removesuffix('.pickle')
    #plot_filename = f'./PE_mdc/inj_plots/heatmap_{label}.png' 
    #plot_filename = f'./PE_mdc/Sly_test_plots/heatmap_{label}.png'
    #plot_filename = f'./PE_mdc/plots_all_posterior/heatmap_{label}.png'
    #plot_filename = f'./PE_mdc/PE_later_MDC_plots/heatmap_{label}.png'
    plot_filename = f'./mdc_analysis/plots/updated_fits2/heatmaps/heatmap_{label}.png'
    print(f'Saving: {plot_filename}')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close('all')

if __name__ == "__main__":
    #events = ['S220831dl']
    #events = ['S220903o']
    #events = ['S220906w']
    #events = [events[1]]
    #events = ['S220908fc']
    #events = ['S220904gn']
    #events = ['S220830bb']
    #events = ['S220829kl']
    for event in events:
        print(event)
        try:
            lc_prefix = 'PE_lc'
            ej_prefix = 'PE_ejecta'
            filepath = f'{event_path}/{event}'
            files = os.listdir(filepath)
            lc_files = [f for f in files if f.startswith(lc_prefix)]
            ej_files = [f for f in files if f.startswith(ej_prefix)]
        except FileNotFoundError:
            continue
        for lc_filename, ej_filename in zip(lc_files, ej_files):
            heat_map(filepath, lc_filename, ej_filename, event)
            #try: heat_map(filepath, lc_filename, ej_filename, event)
            #except:
            #    plt.close('all')
            #    continue
