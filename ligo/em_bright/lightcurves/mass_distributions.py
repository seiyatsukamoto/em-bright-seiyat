"""Module containing intial compoment mass distribution functions
"""
import numpy as np
import scipy.stats as stats
from .lightcurve_utils import alsing_dist, farrow_dist, zhu_dist


def NSBH_uniform(mass_draws):
    m1 = 3*np.ones(mass_draws)+5*np.random.rand(mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def BNS_uniform(mass_draws):
    m1 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def BNS_alsing(mass_draws):
    # From Alsing 2018 (arxiv 1709.07889)
    a_dist = alsing_dist(a=1.1, b=2.8)
    m1 = a_dist.rvs(size=mass_draws)
    m2 = a_dist.rvs(size=mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def BNS_farrow(mass_draws):
    # From Farrow 2019 (arxiv 1902.03300)
    f_dist = farrow_dist(a=1.1, b=2.8)
    m1 = f_dist.rvs(size=mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def NSBH_zhu(mass_draws):
    # From Zhu 2021 (arxiv 2011.02717)
    z_dist = zhu_dist(a=2.8, b=25)
    dist_NS_zhu = stats.norm(1.33, scale=.01)
    m1 = z_dist.rvs(size=mass_draws)
    m2 = dist_NS_zhu.rvs(size=mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def NSBH_LRR(mass_draws):
    # From Living Reviews in Relativity
    ns_astro_mass_dist = stats.norm(1.33, 0.09)
    bh_astro_mass_dist = stats.pareto(b=1.3)
    m1 = bh_astro_mass_dist.rvs(size=mass_draws)
    m2 = ns_astro_mass_dist.rvs(size=mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def BNS_LRR(mass_draws):
    # From Living Reviews in Relativity
    ns_astro_mass_dist = stats.norm(1.33, 0.09)
    m1 = ns_astro_mass_dist.rvs(size=mass_draws)
    m2 = ns_astro_mass_dist.rvs(size=mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type
