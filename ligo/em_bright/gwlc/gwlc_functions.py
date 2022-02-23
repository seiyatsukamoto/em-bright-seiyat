"""Module containing functions to calculate mass ejecta and lightcurves from
   initial component masses
"""
import numpy as np
import scipy.stats as stats
from .gwlc_utils import alsing_dist, farrow_dist, zhu_dist

np.random.seed(0)


def initial_mass_draws(Type, mass_draws):
    '''Draws component masses (NS's or BH's) from the desired distribution

       Parameters
       ----------
       Type: str
           one of the below component mass dists
       mass_draws: int
           number of component mass pairs to draw

       Returns
       -------
       m1_sorted: numpy array
           more massive component mass
       m2_sorted: numpy array
           less massive component mass
       thetas: numpy array
           inclination or viewing angle
       MergerType: str
           can be BNS, NSBH, or NotSpecified
    '''

    if Type == 'NSBH_uniform':
        m1 = 3*np.ones(mass_draws)+5*np.random.rand(mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'NSBH'

    elif Type == 'BNS_uniform':
        m1 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'BNS'

    elif Type == 'BNS_alsing':
        # From Alsing 2018 (arxiv 1709.07889)
        a_dist = alsing_dist(a=1.1, b=2.8)
        m1 = a_dist.rvs(size=mass_draws)
        m2 = a_dist.rvs(size=mass_draws)
        MergerType = 'BNS'

    elif Type == 'BNS_farrow':
        # From Farrow 2019 (arxiv 1902.03300)
        f_dist = farrow_dist(a=1.1, b=2.8)
        m1 = f_dist.rvs(size=mass_draws)
        m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
        MergerType = 'BNS'

    elif Type == 'NSBH_zhu':
        # From Zhu 2021 (arxiv 2011.02717)
        z_dist = zhu_dist(a=2.8, b=25)
        dist_NS_zhu = stats.norm(1.33, scale=.01)
        m1 = z_dist.rvs(size=mass_draws)
        m2 = dist_NS_zhu.rvs(size=mass_draws)
        MergerType = 'NSBH'

    elif Type == 'NSBH_LRR':
        ns_astro_mass_dist = stats.norm(1.33, 0.09)
        bh_astro_mass_dist = stats.pareto(b=1.3)
        m1 = bh_astro_mass_dist.rvs(size=mass_draws)
        m2 = ns_astro_mass_dist.rvs(size=mass_draws)
        MergerType = 'NSBH'

    elif Type == 'BNS_LRR':
        ns_astro_mass_dist = stats.norm(1.33, 0.09)
        m1 = ns_astro_mass_dist.rvs(size=mass_draws)
        m2 = ns_astro_mass_dist.rvs(size=mass_draws)
        MergerType = 'BNS'

    # sort to make sure m1 > m2
    for i, (m_a, m_b) in enumerate(zip(m1, m2)):
        if m_a < m_b:
            m1[i], m2[i] = m_b, m_a

    return m1, m2, MergerType
