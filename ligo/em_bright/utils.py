# Copyright (C) 2018 Shaon Ghosh, Shasvath Kapadia, Deep Chatterjee
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


import os
import pickle

from argparse import ArgumentParser
from configparser import ConfigParser
import glob
import sqlite3

import numpy as np
import pandas as pd
from astropy.table import Column, Table, vstack
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier


def join():
    """
    Joins the extracted sim-coinc data from GstLAL injection
    campaigns into a single Astropy table
    """
    parser = ArgumentParser(
        "Join extracted GstLAL sim-coinc parameters as astropy table")
    parser.add_argument("-i", "--input", required=True,
                        help="Directory storing extracted sim-coinc files")
    parser.add_argument("-c", "--config", required=True,
                        help="Name of the config file")
    parser.add_argument("-o", "--output", required=True,
                        help="Name of the output file")
    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    extract_prefix = config.get('output_filenames',
                                'em_bright_extract_prefix')
    extract_suffix = config.get('output_filenames',
                                'em_bright_extract_suffix')
    cols = config.get('core',
                      'sqlite_cols')
    list_of_files = glob.glob(
        os.path.join(args.input, '{}*{}'.format(extract_prefix,
                                                extract_suffix)))
    cols = cols.split(',')
    # FIXME delimited '|' is a fragile piece
    data = vstack(
        [Table.read(f, format='ascii', delimiter='|', names=cols)
         for f in list_of_files]
    )
    # FIXME need a better implementation
    # convert injected detector frame mass to source frame masses
    inj_mass1_source = Column(data['inj_m1']/(1.0 + data['inj_redshift']),
                              name='inj_mass1_source_frame')
    inj_mass2_source = Column(data['inj_m2']/(1.0 + data['inj_redshift']),
                              name='inj_mass2_source_frame')
    ID = Column(np.arange(len(data)), name='id')
    data.add_column(ID, index=0)
    data.add_column(inj_mass1_source, index=0)
    data.add_column(inj_mass2_source, index=0)

    data.write(args.output, format='ascii', delimiter='\t')


def extract():
    """
    Ingests a GstLAL injection campaign sqlite database and
    outputs the list of coinc parameters in a flat file
    """
    parser = ArgumentParser(
        "Get sim-coinc maps for LIGO GstLAL injection sqlite database")
    parser.add_argument("-i", "--input", required=True,
                        help="sqlite database")
    parser.add_argument("-o", "--output", required=True,
                        help="Output file, stored as numpy array")
    args = parser.parse_args()

    cur = sqlite3.connect(args.input).cursor()
    cur.execute(
        """
        CREATE TEMPORARY TABLE
        sim_coinc_map_helper
        AS
        SELECT a.event_id as sid,
        coinc_event.coinc_event_id as cid,
        coinc_event.likelihood as lr
        FROM coinc_event_map as a
        JOIN coinc_event_map AS b ON (b.coinc_event_id == a.coinc_event_id)
        JOIN coinc_event ON (coinc_event.coinc_event_id == b.event_id)
        WHERE a.table_name == 'sim_inspiral'
        AND b.table_name == 'coinc_event'
        AND NOT EXISTS (
            SELECT * FROM time_slide WHERE
            time_slide.time_slide_id == coinc_event.time_slide_id
            AND time_slide.offset != 0
        )"""
    )

    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS
        sim_coinc_map_helper_index ON sim_coinc_map_helper (sid, cid)
        """
    )

    cur.execute(
        """
        CREATE TEMPORARY TABLE
            sim_coinc_map
        AS
            SELECT
                sim_inspiral.simulation_id AS simulation_id,
                (
                    SELECT cid FROM
                                sim_coinc_map_helper
                    WHERE sid = simulation_id
                    ORDER BY lr DESC
                    LIMIT 1
                ) AS coinc_event_id
            FROM sim_inspiral
            WHERE coinc_event_id IS NOT NULL
        """
    )

    cur.execute("""DROP INDEX sim_coinc_map_helper_index""")

    query = """
    SELECT
    sim_inspiral.mass1,
    sim_inspiral.mass2,
    sim_inspiral.spin1z,
    sim_inspiral.spin2z,
        sim_inspiral.alpha3,
    sngl_inspiral.mass1,
    sngl_inspiral.mass2,
    sngl_inspiral.spin1z,
    sngl_inspiral.spin2z,
    sngl_inspiral.Gamma1,
    coinc_inspiral.combined_far,
    coinc_inspiral.snr,
    coinc_inspiral.end_time
    FROM
    sim_coinc_map
    JOIN
    sim_inspiral
    ON
    sim_coinc_map.simulation_id==sim_inspiral.simulation_id
    JOIN
    coinc_event_map
    ON
    sim_coinc_map.coinc_event_id == coinc_event_map.coinc_event_id
    JOIN
    coinc_inspiral
    ON
    sim_coinc_map.coinc_event_id == coinc_inspiral.coinc_event_id
    JOIN
    sngl_inspiral
    ON (
        coinc_event_map.table_name == 'sngl_inspiral'
        AND coinc_event_map.event_id == sngl_inspiral.event_id
    )
    WHERE
    sngl_inspiral.ifo=='H1';
    """
    np.savetxt(args.output, np.array(cur.execute(query).fetchall()),
               fmt='%f|%f|%f|%f|%f|%f|%f|%f|%f|%d|%e|%f|%f')


def train():
    parser = ArgumentParser(
        description='Executable to train source classifier from injections')

    parser.add_argument(
        '-i', '--input',
        help='Pickled dataframe containing source categorized data')
    parser.add_argument(
        '-o', '--output',
        help='Pickled object storing the trained classifiers')
    parser.add_argument(
        '-d', '--param-sweep-plot', action='store_true',
        help='Supply to obtain a parameter sweep')
    parser.add_argument(
        '-c', '--config', required=True,
        help='Config file with additional parameters')

    args = parser.parse_args()

    config = ConfigParser()
    config.read(args.config)
    # compulsory sections in config
    required_sections = ['core',
                         'em_bright']
    assert all(config.has_section(s) for s in required_sections), \
        'Config file must have sections %s' % (required_sections,)

    # get column names and values from config
    feature_cols = config.get('em_bright',
                              'feature_cols').split(',')
    category_cols = config.get('em_bright',
                               'category_cols').split(',')
    threshold_cols = config.get('em_bright',
                                'threshold_cols').split(',')
    all_cols = feature_cols + category_cols + threshold_cols

    threshold_values = map(
        eval, config.get(
            'em_bright',
            'threshold_values').split(',')
        )
    threshold_type = config.get('em_bright',
                                'threshold_type').split(',')
    # read dataframe, check sanity
    with open(args.input, 'rb') as f:
        df = pickle.load(f)

    assert all(col in df.keys() for col in all_cols), \
        'Dataframe must contain columns %s' % (all_cols,)

    # create masked array based on threshold values,
    # extract features and targets
    mask = np.ones(len(df)).astype(bool)
    for col, value, typ in zip(threshold_cols, threshold_values,
                               threshold_type):
        mask &= df[col] < value if typ == 'lesser' else \
            df[col] > value if typ == 'greater' else True
    features = df[feature_cols][mask]
    targets = df[category_cols][mask]
    # train Nearest Neighbor classifier for each category
    clf_kwargs = eval(config.get('em_bright',
                                 'clf_kwargs'))
    clfs = []
    for category, target_value in targets.iteritems():
        # run KFold cross-validation
        res_predict, res_predict_proba = run_k_fold_split(features,
                                                          target_value,
                                                          **clf_kwargs)
        test_results = pd.DataFrame(
            np.vstack((features.T, target_value,
                       res_predict, res_predict_proba)).T,
            columns=feature_cols + ['targets', 'predict', 'predict_proba']
        )
        test_results_filename = 'cross-val-res-' + category + '.csv'
        test_results.to_csv(test_results_filename, index=False)
        # train on the full dataset
        clf = KNeighborsClassifier(**clf_kwargs)
        clf.fit(features, target_value)
        if args.param_sweep_plot:
            _create_param_sweep_plot(clf, category)
        clfs.append(clf)
    # append the output filename of the classifier
    clfs.extend([args.output])
    with open(args.output, 'wb') as f:
        pickle.dump(clfs, f)


def _create_param_sweep_plot(clf, category):
    """Create a fake recovered parameter space and plot
    the predictions for the classifier sweeping across
    masses.
    """
    import matplotlib.pyplot as plt
    mass1 = np.linspace(1, 100, 1000)
    mass2 = np.linspace(1, 100, 1000)
    t = Table(
        data=np.vstack(
            (np.repeat(mass1, mass2.size),
             np.tile(mass2, mass1.size))
        ).T, names=('mass1', 'mass2')
    )
    mask = t['mass1'] > t['mass2']
    t = t[mask]
    spins = Table(
        data=np.vstack(
            (np.repeat(np.linspace(0, 1, 2), 2),
             np.tile(np.linspace(0, 1, 2), 2))
        ).T,
        names=('chi1', 'chi2')
    )

    fig = plt.figure(figsize=(14, 20))
    for idx, spin_vals in enumerate(spins):
        SNR = 10.
        title = "chi1 = {0}; chi2 = {1}; SNR = {2}".format(spin_vals['chi1'],
                                                           spin_vals['chi2'],
                                                           SNR)
        SNR *= np.ones(np.sum(mask))
        chi1 = spin_vals['chi1'] * np.ones(np.sum(mask))
        chi2 = spin_vals['chi2'] * np.ones(np.sum(mask))

        # make predictions and make plots
        param_sweep_features = np.stack(
            [t['mass1'], t['mass2'], chi1, chi2, SNR]
        ).T
        predictions = clf.predict_proba(param_sweep_features).T[1]
        # plot against m1-m2 the non-zero p-values
        make_plots(param_sweep_features, predictions, title, (fig, idx+1))
    plt.savefig('param_sweep_'+category+'.png')


def make_plots(features, predictions, title, fig_idx):
    import matplotlib.pyplot as plt
    fig_, idx = fig_idx
    fig_.add_subplot(4, 1, idx)
    # indices 0 and 1 correspond to mass1 and mass2 respectively
    plt.scatter(features.T[0], features.T[1],
                s=10, c=predictions)
    plt.title(title)
    plt.tight_layout()
    plt.colorbar(label='Probability')
    # plot chirp mass contours
    m1 = np.linspace(1, 50, 100)
    m2 = np.linspace(1, 50, 100)
    M1, M2 = np.meshgrid(m1, m2)
    Mc = (M1*M2)**(3./5.)/(M1 + M2)**(1./5.)
    Mc = np.tril(Mc).T

    CS = plt.contour(
        M1, M2, Mc, levels=[5, 6, 7, 8, 9],
        colors='black', linewidths=1
    )

    plt.clabel(CS, inline=True, fontsize=16)
    plt.xlim((1, 50))
    plt.ylim((1, 14))
    plt.axhline(y=3.0, c='r')
    plt.xlabel(r'$m_1$', fontsize=16)
    plt.ylabel(r'$m_2$', fontsize=16)
    plt.title(title)


def run_k_fold_split(features, targets, n_splits=10, random_state=0,
                     **kwargs):
    """Split the `features` in `n_splits`, train on `n_splits - 1`
    sets, test on the last fraction. This performed across the complete
    dataset.

    Parameters
    ----------
    features : numpy.ndarray
        Feature set
    targets : numpy.ndarray
        Target set containing binary classification of `features`
    n_splits : int
        Number of splits for `features` and `targets`
    random_state : int
        Random seef for the split
    **kwargs
        Keyword arguments passed to `KNearestNeighborClassifier`

    Returns
    -------
    tuple
        (res_predict, res_predict_proba) - predictions and probabilities
    """
    res_predict_proba = np.empty(len(features))
    res_predict = np.empty(len(features))
    res_predict_proba[:] = np.nan
    res_predict[:] = np.nan

    sss = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sss.get_n_splits(features, targets)
    for train_index, test_index in sss.split(features, targets):
        X_train, X_test, y_train = \
            features.iloc[train_index], \
            features.iloc[test_index], \
            targets.iloc[train_index]

        predict_proba, predict = run_KNN_classifier(X_train, y_train, X_test,
                                                    **kwargs)
        # second column is the prob of NS/EMB
        predict_proba = predict_proba.T[1]

        res_predict_proba[test_index] = predict_proba
        res_predict[test_index] = predict
    return res_predict, res_predict_proba


def run_KNN_classifier(X_train, y_train,
                       X_test,
                       **kwargs):
    '''
    Run KNearestNeighborClassifier classifier returns
    `clf.predict_proba`

    Parameters
    ----------
    X_train : numpy.ndarray
        Feature training set, can be array or Dataframe
    y_train : numpy.array
        Target training set, 0 or 1 binary classification
    X_test : numpy.ndarray
        Feature testing set, can be array or DataFrame
    '''
    clf = KNeighborsClassifier(**kwargs)
    clf.fit(X_train, y_train)
    predictions_proba = clf.predict_proba(X_test)
    predict = clf.predict(X_test)
    return predictions_proba, predict
