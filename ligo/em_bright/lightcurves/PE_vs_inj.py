import os
import numpy as np
import astropy
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import pickle5 as pickle
from gwemlightcurves import lightcurve_utils
from astropy.table import Table

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_with_RP'
#inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/SLy_injs'
# event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/PE_later_MDC'
# inj_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/PE_mdc/Sly_injs_later_mdcs'

#event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_MDC9_1000'
event_path = '/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/PE_q_MDC9_1000'
inj_path = f'/home/andrew.toivonen/em-bright-andrew/ligo/em_bright/lightcurves/mdc_analysis/SLy_MDC9'

events = os.listdir(inj_path)
#events = events[:20]

def PE_vs_inj(PEs, injs, quantity, mc_inj, mc_mins, mc_maxs, all_PEs, all_injs):
    '''
    '''

    plt.figure(figsize=(8,6))
    plt.scatter(mc_inj, injs)
    plt.xlabel(f'mchirp inj')
    plt.ylabel(f'q inj')
    #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/mc_vs_q_inj_{quantity}.png')
    plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/mc_vs_q_inj_{quantity}.png')
    plt.close()

    mc_inj, injs = np.hstack(mc_inj), np.hstack(injs)
    mc_inj_plot, injs_plot = mc_inj[mc_inj < 2], injs[mc_inj < 2]
    mc_inj_plot, injs_plot = mc_inj_plot[mc_inj_plot > 1], injs_plot[mc_inj_plot > 1]

    plt.figure(figsize=(8,6))
    plt.hist2d(mc_inj, injs, bins=[np.linspace(1,2,15), np.linspace(0,1,15)])
    plt.xlabel(f'mchirp inj')
    plt.ylabel(f'q inj')
    #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/mc_vs_q_inj_hist_{quantity}.png')
    plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/mc_vs_q_inj_hist_{quantity}.png')
    plt.close()

    print(len(injs))
    plt.figure(figsize=(8,6))
    plt.scatter(injs, PEs)
    plt.plot([np.min(injs), np.max(injs)], [np.min(injs), np.max(injs)], color='black')
    plt.xlabel(f'{quantity} inj')
    plt.ylabel(f'{quantity} median PE')
    #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/PE_vs_inj_{quantity}.png')
    plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/PE_vs_inj_{quantity}.png')
    plt.close()

    #print(all_PEs)
    #print(np.hstack(all_PEs))
    all_PEs = np.hstack(all_PEs)
    all_injs = np.hstack(all_injs)

    #np.array(all_PEs).flatten()
    #np.array(all_injs).flatten()

    all_injs_plot, all_PEs_plot = all_injs[all_PEs < 2], all_PEs[all_PEs < 2]
    all_injs_plot, all_PEs_plot = all_injs_plot[all_PEs_plot > 1], all_PEs_plot[all_PEs_plot > 1]

    plt.figure(figsize=(8,6))
    #plt.scatter(all_injs, all_PEs)
    plt.plot([np.min(injs), np.max(injs)], [np.min(injs), np.max(injs)], color='black')
    #plt.scatter(injs, PEs, label='Median PE')
    plt.hist2d(all_injs, all_PEs, bins=30)
    plt.xlabel(f'{quantity} inj')
    plt.ylabel(f'{quantity} PE')
    #plt.legend()
    #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/all_PE_vs_inj_{quantity}.png')
    plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/all_PE_vs_inj_{quantity}.png')
    plt.close()

    if quantity == 'q':
        filename = '/home/andrew.toivonen/farah/O1O2O3all_mass_h_iid_mag_iid_tilt_powerlaw_redshift_maxP_events_all.h5'
        RP_events = Table.read(filename)
        m1, m2 = RP_events['mass_1'], RP_events['mass_2']
        RP_events['mchirp'] = ((m1*m2)**(3./5.)) * ((m1 + m2)**(-1./5.))
        RP_mc, RP_q = RP_events['mchirp'], m2/m1

        #RP_mc_plot, RP_q_plot = RP_mc[RP_mc < 2], RP_q[RP_mc < 2]
        #RP_mc_plot, RP_q_plot = RP_mc_plot[RP_mc_plot > 1], RP_q_plot[RP_mc_plot > 1]

        plt.figure(figsize=(8,6))
        plt.scatter(RP_mc, RP_q)
        plt.xlabel(f'Farah mchirp')
        plt.ylabel(f'Farah q')
        plt.xlim([1,2])
        #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/mc_vs_q_RP.png')
        plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/mc_vs_q_farah.png')
        plt.close()
        plt.figure(figsize=(8,6))
        plt.hist2d(RP_mc, RP_q, bins=[np.linspace(1,2,15), np.linspace(0,1,15)])
        plt.xlabel(f'Farah mchirp')
        plt.ylabel(f'Farah q')
        #plt.xlim([1,2])
        #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/mc_vs_q_RP_hist.png')
        plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/mc_vs_q_farah_hist.png')
        plt.close()

        NSmin = 1.2768371655
        #NSmax = 1.4402725191
        #max_mc, min_mc = np.max(mcs), np.min(mcs)
        q_medians = []
        q_mins, q_maxs = [], []
        for min_mc, max_mc in zip(mc_mins, mc_maxs):
            RP_events = RP_events[RP_events['mass_1'] >= NSmin]
            RP_events = RP_events[RP_events['mass_2'] >= NSmin]

            RP_events_cut = RP_events[RP_events['mchirp'] >= min_mc]
            RP_events_cut = RP_events_cut[RP_events_cut['mchirp'] <= max_mc]
            RP_events_cut = RP_events[RP_events['mchirp'] >= .9*min_mc]
            RP_events_cut = RP_events_cut[RP_events_cut['mchirp'] <= 1.1*max_mc]

            if len(RP_events_cut) > 0:
                q_draws = np.random.choice(RP_events_cut['mass_ratio'], 50)
            q_medians.append(np.median(q_draws))
            q_mins.append(np.min(q_draws))
            q_maxs.append(np.max(q_draws))
            #eta_draws = lightcurve_utils.q2eta(q_draws)
            #m1_draws, m2_draws = lightcurve_utils.mc2ms(PE_data['mchirp'].values, eta_draws)
        plt.figure(figsize=(8,6))
        plt.scatter(injs, q_medians)
        plt.plot([np.min(injs), np.max(injs)], [np.min(injs), np.max(injs)], color='black')
        plt.xlabel(f'{quantity} inj')
        plt.ylabel(f'{quantity} median Farah draws')
        #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/RP_vs_inj_{quantity}.png')
        plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/farah_vs_inj_{quantity}.png')
        plt.close()
        plt.figure(figsize=(8,6))
        plt.scatter(injs, q_mins, label='Min of draws')
        plt.scatter(injs, q_maxs, label='Max of draws')
        plt.plot([np.min(injs), np.max(injs)], [np.min(injs), np.max(injs)], color='black')
        plt.xlabel(f'{quantity} inj')
        plt.ylabel(f'{quantity} Farah draws')
        plt.legend()
        #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/RP_range_vs_inj_{quantity}.png')
        plt.savefig(f'paper_plots/PE_vs_inj_plots_MDC9/farah_range_vs_inj_{quantity}.png') 
        plt.close()
   
    plt.figure(figsize=(8,6))
    plt.hist(np.array(injs), bins=20)
    #plt.plot([np.min(injs), np.max(injs)], [np.min(injs), np.max(injs)], color='black')
    plt.xlabel(f'{quantity} inj')
    plt.ylabel('Counts')
    #plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/inj_hist_{quantity}.png')
    plt.savefig(f'paper_plots/PE_vs_inj_plots_later_mdc/inj_hist_{quantity}.png')
    plt.close()

def PP_percentiles(data, inj):
    '''
    '''
    per = percentileofscore(data, inj)

def fetch_PE_vs_inj(quantity):
    #events = [events[1]]
    #events = ['S220908fc']
    #events = ['S220904gn']
    #events = ['S220830bb']
    percentiles = []
    injs = []
    all_injs = []
    PEs = []
    all_PEs = []
    mc_mins = []
    mc_maxs = []
    mc_inj = []
    #quantity = 'mej'
    quantity = 'mchirp'
    #quantity = 'm1'
    #quantity = 'm2'
    #quantity = 'q'
    #quantity = 'theta'
    print(f'running for {quantity}')
    for event in events:
        print(event)
        try:
            filepath = f'{event_path}/{event}'
            files = os.listdir(filepath)
            prefix = 'PE_ejecta'
            files = [f for f in files if f.startswith(prefix)]
        except FileNotFoundError:
            continue
        for filename in files:
            print(filename)
            try:
                path = f'{filepath}/{filename}'
                with open (path, 'rb') as f:
                    data = pickle.load(f)

                try:
                    inj = f'{inj_path}/{event}/inj_lc_1_table_{event}_SLy_1x1_1.0.pickle'
                    with open (inj, 'rb') as f:
                        inj_data = pickle.load(f)
                except FileNotFoundError:
                    inj = f'{inj_path}/{event}/inj_lc_table_{event}_SLy_1x1_1.0.pickle'
                    with open (inj, 'rb') as f:
                        inj_data = pickle.load(f)
                #with open (inj, 'rb') as f:
                #    inj_data = pickle.load(f)
                #percentiles.append(percentileofscore(data[quantity], inj_data[quantity]))
                injs.append(inj_data[quantity])
                all_injs.append(inj_data[quantity].data*np.ones(len(data[quantity])))
                PEs.append(np.median(data[quantity]))
                all_PEs.append(data[quantity].data)
                mc_inj.append(inj_data['mchirp'])
                mc_mins.append(np.min(data['mchirp']))
                mc_maxs.append(np.max(data['mchirp']))
                print(data[quantity].data)
            except FileNotFoundError:
                continue
    #all_inj = astropy.table.vstack(all_injs)
    #all_PEs = astropy.table.vstack(all_PEs)
    return PEs, injs, quantity, mc_inj, mc_mins, mc_maxs, all_PEs, all_injs

if __name__ == "__main__":
    PEs, injs, quantity, mc_inj, mc_mins, mc_maxs, all_PEs, all_injs = fetch_PE_vs_inj('q')
    PE_vs_inj(PEs, injs, quantity, mc_inj, mc_mins, mc_maxs, all_PEs, all_injs)
