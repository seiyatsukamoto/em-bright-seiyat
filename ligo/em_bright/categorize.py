# Copyright (C) 2020 Shaon Ghosh, Deep Chatterjee
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


import argparse
import functools
import os

import numpy as np
import pandas as pd
import pickle

from . import computeDiskMass, EOS_BAYES_FACTORS, EOS_MAX_MASS


class _TupleHandler(object):
    def __call__(self, func):
        @functools.wraps(func)
        def handle(*args, **kwargs):
            try:
                (infile, outfile, max_mass, eosname), *_ = args
            except (ValueError, TypeError):
                return func(*args, **kwargs)
            return func(infile, outfile, max_mass, eosname)
        return handle


def regularize(m1, m2, chi1, chi2):
    '''Transform masses based on the convention m1 > m2.
    Apply the same for spins.
    '''
    # Find cases where the mass2 is greater than the mass1
    largerMass1 = m1 >= m2
    largerMass2 = m2 > m1

    # Set up vectors to store the regularized masses and spins
    m_primary = np.zeros_like(m1)
    m_secondary = np.zeros_like(m2)
    chi_primary = np.zeros_like(chi1)
    chi_secondary = np.zeros_like(chi2)

    # Swap vales of masses if mass1 < mass2
    m_primary[largerMass1] = m1[largerMass1]
    m_primary[~largerMass1] = m2[largerMass2]
    m_secondary[~largerMass2] = m2[~largerMass2]
    m_secondary[largerMass2] = m1[~largerMass1]

    # Assign the spin values accordingly
    chi_primary[largerMass1] = chi1[largerMass1]
    chi_primary[~largerMass1] = chi2[~largerMass1]
    chi_secondary[~largerMass2] = chi2[~largerMass2]
    chi_secondary[largerMass2] = chi1[~largerMass1]

    return (m_primary, m_secondary, chi_primary, chi_secondary)


@_TupleHandler()
def embright_categorization(inFile, outFile, eosname='2H', mNs_mass=None):
    '''Categorize whether the binary has a neutron star, and any non-zero
    remnant post merger, based on the ``eosname``.

    Parameters
    ----------
    inFile : str
        input filename
    outFile : str
        output filename
    eosname : str
        neutron star equation of state. Assumed implemented in lalsimulation.
        Default 2H.
    mNS_mass : float
        Provide to override the maximum mass from ``eosname``.
    '''
    df = pd.read_table(inFile, delimiter='\t')
    m1_inj, m1_rec = df.inj_m1.values, df.rec_m1.values
    m2_inj, m2_rec = df.inj_m2.values, df.rec_m2.values
    redshift_inj = df.inj_redshift.values
    m1_inj_source = m1_inj / (1 + redshift_inj)
    m2_inj_source = m2_inj / (1 + redshift_inj)

    chi1_inj, chi1_rec = df.inj_spin1z.values, df.rec_spin1z.values
    chi2_inj, chi2_rec = df.inj_spin2z.values, df.rec_spin2z.values
    m1_inj, m2_inj, chi1_inj, chi2_inj = regularize(
        m1_inj, m2_inj, chi1_inj, chi2_inj
    )
    m1_rec, m2_rec, chi1_rec, chi2_rec = regularize(
        m1_rec, m2_rec, chi1_rec, chi2_rec
    )
    m1_inj_source, m2_inj_source, _, _ = regularize(
        m1_inj_source, m2_inj_source, chi1_inj, chi2_inj
    )  # spins regularized from previous step.
    if mNs_mass:
        NS_classified = m2_inj_source < mNs_mass
    else:
        NS_classified = m2_inj_source < EOS_MAX_MASS[eosname]
    NS_classified = NS_classified.astype(int)

    R_isco_hat_inj = computeDiskMass.compute_isco(chi1_inj)
    R_isco_hat_rec = computeDiskMass.compute_isco(chi1_rec)

    Compactness_inj, *_ = computeDiskMass.computeCompactness(m2_inj,
                                                             eosname=eosname)
    Compactness_rec, *_ = computeDiskMass.computeCompactness(m2_rec,
                                                             eosname=eosname)

    mc_inj = (m1_inj*m2_inj)**(3/5)/((m1_inj + m2_inj)**(1/5))
    mc_rec = (m1_rec*m2_rec)**(3/5)/((m1_rec + m2_rec)**(1/5))

    frac_mc_err = np.abs(mc_inj - mc_rec)/mc_inj

    q_inj = m1_inj/m2_inj
    q_rec = m1_rec/m2_rec

    disk_mass_inj = computeDiskMass.computeDiskMass(
        m1_inj_source, m2_inj_source, chi1_inj,
        chi2_inj, eosname=eosname
    )
    EMB_classified = disk_mass_inj > 1e-3
    EMB_classified = EMB_classified.astype(int)

    output = np.vstack(
        (m1_inj, m2_inj, chi1_inj, chi2_inj, redshift_inj,
         mc_inj, q_inj, R_isco_hat_inj, Compactness_inj,
         m1_rec, m2_rec, chi1_rec, chi2_rec,
         mc_rec, frac_mc_err, q_rec, R_isco_hat_rec, Compactness_rec,
         df.cfar.values, df.snr.values, df.gpstime.values, NS_classified,
         EMB_classified)
    ).T

    df_complete = pd.DataFrame(output, columns=[
        'm1_inj', 'm2_inj', 'chi1_inj', 'chi2_inj', 'inj_redshift',
        'mc_inj', 'q_inj', 'R_isco_inj', 'Compactness_inj',
        'm1_rec', 'm2_rec', 'chi1_rec', 'chi2_rec', 'mc_rec',
        'frac_mc_err', 'q_rec', 'R_isco_rec', 'Compactness_rec',
        'cfar', 'snr', 'gpstime', 'NS', 'EMB']
    )
    # drop NSs above max-mass
    try:
        max_mass = EOS_MAX_MASS[eosname]
    except KeyError:
        raise NotImplementedError
    drop_mask = (m1_inj_source > max_mass) & (m1_inj_source < 3.0)  # noqa: E501; Restricts m2 by construction
    df_complete = df_complete.loc[~drop_mask]
    with open(outFile, 'wb') as f:
        pickle.dump(df_complete, f)
    return df_complete


@_TupleHandler()
def mass_gap_categorization(inFile, outFile):
    '''Categorize whether the binary is in the lower mass gap
    i.e., any component is between 3 and 5 solar mass

    Parameters
    ----------
    inFile : str
        input filename
    outFile : str
        output filename
    '''
    df = pd.read_table(inFile, delimiter='\t')
    m1_inj, m1_rec = df.inj_m1.values, df.rec_m1.values
    m2_inj, m2_rec = df.inj_m2.values, df.rec_m2.values
    redshift_inj = df.inj_redshift.values
    m1_inj_source = m1_inj / (1 + redshift_inj)
    m2_inj_source = m2_inj / (1 + redshift_inj)

    chi1_inj, chi1_rec = df.inj_spin1z.values, df.rec_spin1z.values
    chi2_inj, chi2_rec = df.inj_spin2z.values, df.rec_spin2z.values
    m1_inj, m2_inj, chi1_inj, chi2_inj = regularize(
        m1_inj, m2_inj, chi1_inj, chi2_inj
    )
    m1_rec, m2_rec, chi1_rec, chi2_rec = regularize(
        m1_rec, m2_rec, chi1_rec, chi2_rec
    )
    m1_inj_source, m2_inj_source, _, _ = regularize(
        m1_inj_source, m2_inj_source, chi1_inj, chi2_inj
    )  # spins regularized from previous step.
    massgap_classified = (m1_inj_source > 3.0) & (m1_inj_source < 5.0)
    massgap_classified += (m2_inj_source > 3.0) & (m2_inj_source < 5.0)
    output = np.vstack(
        (m1_inj_source, m2_inj_source, m1_inj, m2_inj, chi1_inj,
         chi2_inj, redshift_inj, m1_rec, m2_rec, chi1_rec, chi2_rec,
         df.cfar.values, df.snr.values, df.gpstime.values,
         massgap_classified)
    ).T

    df_complete = pd.DataFrame(output, columns=[
        'm1_inj_source', 'm2_inj_source', 'm1_inj', 'm2_inj', 'chi1_inj',
        'chi2_inj', 'inj_redshift', 'm1_rec', 'm2_rec', 'chi1_rec',
        'chi2_rec', 'cfar', 'snr', 'gpstime', 'mass_gap']
    )
    with open(outFile, 'wb') as f:
        pickle.dump(df_complete, f)
    return df_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", action="store", type=str,
                        help="Name of the input file")
    parser.add_argument("-o", "--output", action="store", type=str,
                        help="Name of the output file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-e", "--eosname", default='2H',
        help="Equation of state used compute remnant matter."
    )
    group.add_argument(
        "--mass-gap", required=False, action="store_true",
        default=False, help="Supply for mass gap categorization")

    args = parser.parse_args()

    if args.mass_gap:
        mass_gap_categorization(args.input, args.output)
    else:
        embright_categorization(args.input, args.output,
                                eosname=args.eosname)


def main_all():
    parser = argparse.ArgumentParser(
        "Run categorization on all available EoSs: "
        f"{EOS_BAYES_FACTORS.keys()}\n"
        "Output will be tagged by EoS names."
    )
    parser.add_argument(
        "-i", "--input", action="store", type=str,
        help="Name of the input file"
    )
    parser.add_argument(
        "-d", "--output-directory", type=str,
        help="Output directory"
    )
    parser.add_argument(
        "-o", "--output-prefix", action="store",
        type=str, help="Input prefix "
    )
    parser.add_argument(
        "-p", "--pool", default=1, type=int,
        help="Pool size"
    )
    args = parser.parse_args()

    from multiprocessing import Pool
    eos_args = [
        (
            args.input, os.path.join(
                args.output_directory,
                f"{args.output_prefix}_{eosname}.pkl"
            ),
            EOS_MAX_MASS[eosname],
            eosname
        )
        for eosname in EOS_BAYES_FACTORS
    ]
    with Pool(args.pool) as p:
        p.map(embright_categorization, eos_args)
