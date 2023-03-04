import os
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import pickle5 as pickle
import pandas as pd
import seaborn as sns

#eos = 'SLy'
#eos = 'APR4_EPP'
#eos = 'H4'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_with_RP'

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_later_MDC'  # works
SLy_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'  # works
APR4_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'  # works

#  inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'  # works

#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/SLy_injs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/Sly_injs'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/SLy_injs_updated_eos'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_MDC9_1000'
#inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/{eos}_MDC9'

#------------------------------------------------------------------------------------------------------------
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_MDC9_1000'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_q_MDC9_1000'
#SLy_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/SLy_MDC9'
#APR4_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/APR4_EPP_MDC9'
#------------------------------------------------------------------------------------------------------------

events = os.listdir(SLy_path)
events = events[:50]
#events = events[0,8,13,18]

for event in events:
    #print(event)
    #if event == 'S230106az':
    #if event == 'S221230n':
    #if event == 'S221219c':
    if event == 'S221218gw':
        print('-----------------------------------------------------')
        print(f'{event} with mej error found')

events_temp = []
for n in [1,9,13,19]:
    events_temp.append(events[n])
#events = events_temp

def violin_plot(quantity, events, band=None, multiple=False):
    #events = ['S221221']
    plt.figure(figsize=(12,9))
    event_count = 0
    m1_labels, m2_labels = [], []
    SLy_vals, APR4_vals = [], []
    dfs = []
    count = 0
    for event in events:
        print(f'event number: {count}') 
        count += 1
        percentiles = []
        print(f'running for {quantity}, {event}')
        if quantity == 'mag':
            prefix = 'PE_lc'
        else:
            prefix = 'PE_ejecta'
        #try:
        if True:
            filepath = f'{event_path}/{event}'
            files = os.listdir(filepath)
            filename = [f for f in files if f.startswith(prefix)][0]
        #except FileNotFoundError:
        #    continue
        #for filename in files:
        try:
            path = f'{filepath}/{filename}'
            with open (path, 'rb') as f:
                event_data = pickle.load(f)
            if quantity != 'mag':
                inj_prefix = 'inj_lc'
                SLy_filepath = f'{SLy_path}/{event}'
                SLy_files = os.listdir(SLy_filepath)
                SLy_f = [f for f in SLy_files if f.startswith(inj_prefix)][0]
                APR4_filepath = f'{APR4_path}/{event}'
                APR4_files = os.listdir(APR4_filepath)
                APR4_f = [f for f in APR4_files if f.startswith(inj_prefix)][0]
                SLy_inj = f'{SLy_path}/{event}/inj_lc_1_table_{event}_SLy_1x1_1.0.pickle'
                APR4_inj = f'{APR4_path}/{event}/inj_lc_1_table_{event}_APR4_EPP_1x1_1.0.pickle'
                #APR4_inj = SLy_inj # remove
                with open (SLy_inj, 'rb') as f:
                    SLy_data = pickle.load(f)
                with open (APR4_inj, 'rb') as f:
                    APR4_data = pickle.load(f)

                #m1_labels.append(SLy_data['m1'].data)
                #m2_labels.append(SLy_data['m2'].data)

        except FileNotFoundError:
            continue

        all_event_data = event_data
        event_data = event_data[quantity]

        if np.max(event_data)>50:
            import pdb; pdb.set_trace()

        if event == 'S230106az':
            import pdb; pdb.set_trace()

        print(all_event_data['q'])
        if np.min(np.abs(all_event_data['q']-.4)) < .05:
            import pdb; pdb.set_trace()

        if np.max(event_data)>1e-11 and np.max(event_data)<50:
            plot_data = event_data[event_data>1e-11]
            plt.violinplot(plot_data, positions=[event_count])
            plt.scatter(event_count, SLy_data[quantity], marker='d', s=400, color = 'red')
            plt.scatter(event_count, APR4_data[quantity], marker='*', s=400, color = 'blue')
            event_count+=1

            m1_labels.append(SLy_data['m1'].data)
            m2_labels.append(SLy_data['m2'].data)
            SLy_vals.append(SLy_data[quantity])
            APR4_vals.append(APR4_data[quantity])
            event_dict = {quantity: plot_data, 'event': event, 'm1': SLy_data['m1'][0], 'm2': SLy_data['m2'][0]}
            df = pd.DataFrame(data=event_dict)
            dfs.append(df)

        '''
        # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
        sns.set(style="darkgrid")
        #df = sns.load_dataset('iris')
        print(df.head())

        # specifying the group list as 'order' parameter and plotting
        sns.violinplot(x='event', y='mej', data=df)
        plt.yscale('log')
        plt.ylabel(r'$m_{ej} \:\:\: [M_\odot]$')
        plt.xlabel('Events')
        plt.savefig(f'./mdc_analysis/plots/violin_plots/violin_plot_MDC9_{quantity}.png')

        plt.close()
        '''
        print(np.max(event_data))
        '''
        if np.max(event_data)>1e-11 and np.max(event_data)<50:
            plot_data = event_data[event_data>1e-11]
            plt.violinplot(plot_data, positions=[event_count])
            plt.scatter(event_count, SLy_data[quantity], marker='d', s=400, color = 'red')
            plt.scatter(event_count, APR4_data[quantity], marker='*', s=400, color = 'blue')
            event_count+=1
            print(event_count)
        '''

    #bars = ('A', 'B', 'C', 'D')
    labels = []
    for m1, m2 in zip(m1_labels, m2_labels):
        print(m1, m2)
        m1, m2 = round(m1[0], 2), round(m2[0], 2)
        labels.append(f'{m1}, {m2} '+r'$M_\odot$')
    #bars = ()

    y_pos = np.arange(len(labels))
    plt.xticks(y_pos, labels, rotation=30, fontsize='12', horizontalalignment='right')
    #plt.xticks(y_pos, bars, color='orange', rotation=45, fontweight='bold', fontsize='17', horizontalalignment='right')
    
    plt.subplots_adjust(bottom=0.15) 
    plt.yscale('log')
    plt.ylabel(r'$m_{ej} \:\:\: [M_\odot]$')
    plt.xlabel('Events')
    plt.scatter([],[], label='SLy', marker='d', s=200, color='red')
    plt.scatter([],[], label='APR4', marker='*', s=200, color='blue')
    plt.ylim(1e-4,1e1)
    plt.legend()
    plt.savefig(f'./mdc_analysis/plots/violin_plots/violin_plot_labeled_q_MDC9_{event_count}_events_{quantity}.png')
    plt.close()

    # set a grey background (use sns.set_theme() if seaborn version 0.11.0 or above)
    sns.set(style="darkgrid")
    df = pd.concat(dfs)

    plt.figure(figsize=(8,6))
    ax = sns.violinplot(x='event', y='mej', data=df)
    ax.set_xticklabels(labels)
    x_pos = np.arange(len(labels))
    plt.scatter(x_pos, SLy_vals, label='SLy', marker='d', s=200, color='red')
    plt.scatter(x_pos, APR4_vals, label='APR4', marker='*', s=200, color='blue')
    plt.yscale('log')
    plt.ylim(1e-3,1e-1)
    plt.ylabel(r'$m_{ej} \:\:\: [M_\odot]$')
    plt.xlabel('Events')
    plt.legend(fontsize='medium')
    plt.savefig(f'./mdc_analysis/plots/violin_plots/violin_plot_MDC9_{quantity}.png')

    print(f'saving plot')


quantity = 'mej'
violin_plot(quantity, events)

'''
def violin_plot(percentiles, quantity, band):
    

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
            plt.savefig(f'mdc_analysis/plots/violin_plots/violin_plot_q_MDC9_{eos}_{quantity}.png')
            plt.close()
    if quantity == 'mag':
        if band == 'K':
            #plt.axvline(.90, color='k', linestyle='dashed', label = '90% Credible Interval')
            plt.legend()
            plt.savefig(f'mdc_analysis/plots/violin_plots/violin_plot_q_MDC9_{eos}_{quantity}.png')
            plt.close()

def PP_percentiles(data, inj):
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
        #print(event)
        #try:
        if True:
            filepath = f'{event_path}/{event}'
            files = os.listdir(filepath)
            files = [f for f in files if f.startswith(prefix)]
        #except FileNotFoundError:
            #continue
        for filename in files:
            #try:
            if True:
                path = f'{filepath}/{filename}'
                with open (path, 'rb') as f:
                    event_data = pickle.load(f)
                for line in event_data:
                    if line['m2'] > line['m1']:
                        print(f'q > 1: {event}')
                if quantity != 'mag':
                    inj_prefix = 'inj_lc'
                    inj_filepath = f'{inj_path}/{event}'
                    inj_files = os.listdir(inj_filepath)
                    inj_files = [f for f in inj_files if f.startswith(inj_prefix)]
                    inj_f = inj_files[0]
                    inj = f'{inj_path}/{event}/{inj_f}'
                    print(inj_files)
                else:
                    inj = f'{inj_path}/{event}/inj_lc_1_table_{event}_{eos}_1x1_1.0.pickle'
                with open (inj, 'rb') as f:
                    inj_data = pickle.load(f)
                if quantity == 'mag':
                    lc_dict, lc_dict_inj = {}, {}
                    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

                    for band_dict in bands:
                        lc_dict[band_dict], lc_dict_inj[band_dict] = [], []
                    #lc_dict[band], lc_dict_inj[band] = [], []

                    for line in event_data:
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
                    percentiles.append(percentileofscore(lc_mins, peak_mag_inj))
                else:
                     
                    #percentiles.append(percentileofscore(data[quantity][:], inj_data[quantity])[0])
            #except FileNotFoundError:
            #    continue
    plt.figure(figsize=(8,6))
    plt.hist(percentiles, bins=20)
    plt.xlabel(f'{quantity} Percentile')
    plt.ylabel('Probability')
    plt.savefig(f'mdc_analysis/plots/PP_plots/hist_{quantity}.png')
    plt.close()

    if quantity == 'mag':
        percentiles = list(100-np.array(percentiles))
        plt.figure(figsize=(8,6))
        plt.hist(percentiles, bins=20)
        plt.xlabel(f'{quantity} Percentile')
        plt.ylabel('Probability')
        plt.savefig(f'mdc_analysis/plots/PP_plots/hist_reverse_{quantity}.png')
        plt.close()
    return event_data   


multiple = True
#quantities = ['mag']
#quantities = ['mchirp']
#quantities = ['m1']
#quantities = ['q']
#quantities = ['theta']
quantities = ['mej']

#quantities = ['q', 'mchirp']

plt.figure(figsize=(12,9))
for quantity in quantities:
    lc_dict, lc_dict_inj = {}, {}
    bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K'] 
    if quantity != 'mag':
        bands = [None]
    for band in bands:
        percentile_list = per_for_quantity(quantity, band)
        violin_plot(percentile_list, quantity, band)
if multiple:
    plt.legend()
    plt.savefig(f'mdc_analysis/plots/violin_plots/violin_plot_q_MDC9_multiple.png')
    plt.close()
'''
