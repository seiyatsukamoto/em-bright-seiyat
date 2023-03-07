import os
import numpy as np
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import pickle5 as pickle
import pandas as pd
import seaborn as sns


#------------------------------------------------------------------------------------------------------------
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_MDC9_1000'
#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_q_MDC9_1000'
#SLy_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/SLy_MDC9'
#APR4_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/APR4_EPP_MDC9'
#------------------------------------------------------------------------------------------------------------

event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/PE_MDC9_500'
SLy_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/SLy_MDC9'
APR4_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/updated_fits2/APR4_EPP_MDC9'


events = os.listdir(SLy_path)
#events = events[:50]

'''
# find events
for idx, event in enumerate(events):
    print(idx, event)
    inj_prefix = 'inj_lc'
    SLy_filepath = f'{SLy_path}/{event}'
    SLy_files = os.listdir(SLy_filepath)
    SLy_f = [f for f in SLy_files if f.startswith(inj_prefix)][0]
    SLy_inj = f'{SLy_path}/{event}/inj_lc_1_table_{event}_SLy_1x1_1.0.pickle'
    with open (SLy_inj, 'rb') as f:
        SLy_data = pickle.load(f)
    print(SLy_data['m1'].data, SLy_data['m2'].data)
'''


# 6 S221229gp 1.5 1.5 
# 9 S221225ew 1.4 1.2
# 12 S221215dm 1.4 1.4
# 21 S221230bk
#[1.74196701] [1.15825213]
# 22 S221220w
#[1.88514653] [1.15200433]
# 35 S221218gw
#[1.24375397] [1.24319691]
# 74 S221221eh
#[3.23830387] [1.19979939]
events_temp = []
#events_list = [6,9,12,21,22,74]
#events_list = [12,9,21,22,74]
events_list = [12,9,22,74]
for n in events_list:
    events_temp.append(events[n])
events = events_temp

def violin_plot(quantity, events, band=None, multiple=False):
    plt.figure(figsize=(12,9))
    event_count = 0
    m1_labels, m2_labels = [], []
    SLy_vals, APR4_vals = [], []
    dfs = []
    count = 0
    for event in events:
        print(event)
        print(f'event number: {count}') 
        count += 1
        percentiles = []
        print(f'running for {quantity}, {event}')
        if quantity == 'mag':
            prefix = 'PE_lc'
        else:
            prefix = 'PE_ejecta'

        filepath = f'{event_path}/{event}'
        files = os.listdir(filepath)
        filename = [f for f in files if f.startswith(prefix)][0]
        
        path = f'{filepath}/{filename}'
        with open (path, 'rb') as f:
            event_data = pickle.load(f)
        if quantity == 'mag':
            #quantity_plot == 'peak_r'
            inj_prefix = 'inj_lc'
            SLy_filepath = f'{SLy_path}/{event}'
            SLy_files = os.listdir(SLy_filepath)
            SLy_f = [f for f in SLy_files if f.startswith(inj_prefix)][0]
            APR4_filepath = f'{APR4_path}/{event}'
            APR4_files = os.listdir(APR4_filepath)
            APR4_f = [f for f in APR4_files if f.startswith(inj_prefix)][0]
            SLy_inj = f'{SLy_path}/{event}/inj_lc_1_table_{event}_SLy_1x1_1.0.pickle'
            APR4_inj = f'{APR4_path}/{event}/inj_lc_1_table_{event}_APR4_EPP_1x1_1.0.pickle'
            with open (SLy_inj, 'rb') as f:
                SLy_data = pickle.load(f)
            with open (APR4_inj, 'rb') as f:
                APR4_data = pickle.load(f)
            
            lc_dict, lc_dict_SLy, lc_dict_APR4 = {}, {}, {}
            bands = ['u', 'g', 'r', 'i', 'z', 'y', 'J', 'H', 'K']

            for band in bands:
                lc_dict[band], lc_dict_SLy[band], lc_dict_APR4[band] = [], [], []

            for line in event_data:
                #t = line['t']
                mags = line['mag']
                for idx, mag in enumerate(mags):
                    lc_dict[bands[idx]].append(np.min(mag))

            for line in SLy_data:
                #t = line['t']
                mags = line['mag']
                for idx, mag in enumerate(mags):
                    lc_dict_SLy[bands[idx]].append(np.min(mag))

            for line in APR4_data:
                #t = line['t']
                mags = line['mag']
                for idx, mag in enumerate(mags):
                    lc_dict_APR4[bands[idx]].append(np.min(mag))


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
            with open (SLy_inj, 'rb') as f:
                SLy_data = pickle.load(f)
            with open (APR4_inj, 'rb') as f:
                APR4_data = pickle.load(f)


        all_event_data = event_data
        if quantity == 'mag':
            event_data = lc_dict['r']
            SLy_plot = lc_dict_SLy['r']
            APR4_plot = lc_dict_APR4['r']
        else:
            event_data = event_data[quantity]
            SLy_plot = SLy_data[quantity]
            APR4_plot = APR4_data[quantity]


        plot_data = event_data
        if quantity == 'mej':
            plot_data = event_data[event_data>1e-11]
        plt.violinplot(plot_data, positions=[event_count])
        plt.scatter(event_count, SLy_plot, marker='d', s=400, color = 'red')
        plt.scatter(event_count, APR4_plot, marker='*', s=400, color = 'blue')
        event_count+=1

        m1_labels.append(SLy_data['m1'].data)
        m2_labels.append(SLy_data['m2'].data)
        SLy_vals.append(SLy_plot)
        APR4_vals.append(APR4_plot)
        event_dict = {quantity: plot_data, 'event': event, 'm1': SLy_data['m1'][0], 'm2': SLy_data['m2'][0]}
        df = pd.DataFrame(data=event_dict)
        dfs.append(df)


    labels = []
    for m1, m2 in zip(m1_labels, m2_labels):
        print(m1, m2)
        m1, m2 = round(m1[0], 2), round(m2[0], 2)
        labels.append(f'{m1}, {m2} '+r'$M_\odot$')

    y_pos = np.arange(len(labels))
    plt.xticks(y_pos, labels, rotation=30, fontsize='12', horizontalalignment='right')
    
    plt.subplots_adjust(bottom=0.15) 
    if quantity == 'mej': 
        plt.yscale('log')
        plt.ylabel(r'$m_{ej} \:\:\: [M_\odot]$')
    if quantity == 'mag':
        plt.ylabel(r'Peak Source Frame r $M_{AB}$')
        plt.gca().invert_yaxis()
    plt.xlabel('Injected Source Frame mass 1, mass 2')
    plt.scatter([],[], label='SLy', marker='d', s=200, color='red')
    plt.scatter([],[], label='APR4', marker='*', s=200, color='blue')
    plt.legend()
    plt.savefig(f'./mdc_analysis/plots/updated_fits2/violin_plots/violin_plot_MDC9_{event_count}_events_{quantity}.png')
    plt.close()

    sns.set(style="darkgrid")
    df = pd.concat(dfs)

    plt.figure(figsize=(8,6))
    ax = sns.violinplot(x='event', y=quantity, data=df)
    ax.set_xticklabels(labels)
    x_pos = np.arange(len(labels))
    plt.scatter(x_pos, SLy_vals, label='SLy', marker='d', s=200, color='red')
    plt.scatter(x_pos, APR4_vals, label='APR4', marker='*', s=200, color='blue')
    if quantity == 'mej':
        plt.yscale('log')
        plt.ylabel(r'$m_{ej} \:\:\: [M_\odot]$')
    if quantity == 'mag':
        plt.ylabel(r'Peak Source Frame r $M_{AB}$')
        plt.gca().invert_yaxis()
    plt.xlabel('Injected Source Frame mass 1, mass 2')
    plt.legend(fontsize='medium')
    plt.savefig(f'./mdc_analysis/plots/updated_fits2/violin_plots/violin_plot_MDC9_{quantity}.png')

    print(f'saving plot')


quantity = 'mag'
#quantity = 'mej'
violin_plot(quantity, events)
