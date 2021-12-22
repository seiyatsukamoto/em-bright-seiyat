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

import numpy as np
import pandas as pd
import pickle

from . import computeDiskMass


def regularize(m1, m2, chi1, chi2):
    '''
    This script accepts the values of masses and spins in vector form and
    makes sure that the mass1 is always larger than mass2 also, makes sure
    that the spins of the objects are associated with the right object.
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


def embright_categorization(inFile, outFile, mNs_mass=3.0, eosname='2H'):
    '''
    This script accept the injection-coinc file gotten from the injection
    campaign. It then computes the inference for the EM-Bright and the NS
    classifiers and adds them in the table. It further converts the table
    in pandas DataFrame format and saves into a pickle file
    '''
    df = pd.read_table(inFile, delimiter='\t')
    m1_inj, m1_rec = df.inj_m1.values, df.rec_m1.values
    m2_inj, m2_rec = df.inj_m2.values, df.rec_m2.values
    chi1_inj, chi1_rec = df.inj_spin1z.values, df.rec_spin1z.values
    chi2_inj, chi2_rec = df.inj_spin2z.values, df.rec_spin2z.values
    m1_inj, m2_inj, chi1_inj, chi2_inj = regularize(
        m1_inj, m2_inj, chi1_inj, chi2_inj
    )
    m1_rec, m2_rec, chi1_rec, chi2_rec = regularize(
        m1_rec, m2_rec, chi1_rec, chi2_rec
    )
    NS_classified = m2_inj < mNs_mass

    NS_classified = NS_classified.astype(int)

    R_isco_hat_inj = computeDiskMass.compute_isco(chi1_inj)
    R_isco_hat_rec = computeDiskMass.compute_isco(chi1_rec)

    # For this work we are using the EoS 2H result in the file equil_2H.dat
    Compactness_inj, *_ = computeDiskMass.computeCompactness(m2_inj,
                                                             eosname=eosname)
    Compactness_rec, *_ = computeDiskMass.computeCompactness(m2_rec,
                                                             eosname=eosname)

    mc_inj = (m1_inj*m2_inj)**(3/5)/((m1_inj + m2_inj)**(1/5))
    mc_rec = (m1_rec*m2_rec)**(3/5)/((m1_rec + m2_rec)**(1/5))

    frac_mc_err = np.abs(mc_inj - mc_rec)/mc_inj

    q_inj = m1_inj/m2_inj
    q_rec = m1_rec/m2_rec

    disk_mass_inj = computeDiskMass.computeDiskMass(m1_inj, m2_inj, chi1_inj,
                                                    chi2_inj, eosname=eosname)
    EMB_classified = disk_mass_inj > 0.0
    EMB_classified = EMB_classified.astype(int)

    output = np.vstack(
        (m1_inj, m2_inj, chi1_inj, chi2_inj, df.inj_redshift.values,
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
    with open(outFile, 'wb') as f:
        pickle.dump(df_complete, f)
    return df_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", action="store", type=str,
                        help="Name of the input file")
    parser.add_argument("-o", "--output", action="store", type=str,
                        help="Name of the output file")
    parser.add_argument(
        "-e", "--eosname", default='2H',
        help="Equation of state used compute remnant matter."
    )
    args = parser.parse_args()

    embright_categorization(args.input, args.output,
                            eosname=args.eosname)
