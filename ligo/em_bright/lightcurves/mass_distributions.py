"""Module containing intial compoment mass distribution functions
"""
import numpy as np
import scipy.stats as stats
import ligo.em_bright.lightcurves.lightcurve_utils as lc_utils


def NSBH_uniform(mass_draws):
    '''
    Draws a NS and a BH mass from a uniform distibution

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    m1 = 3*np.ones(mass_draws)+5*np.random.rand(mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def BNS_uniform(mass_draws):
    '''
    Draws NS masses from a uniform distibution

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    m1 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def BNS_alsing(mass_draws):
    '''
    From Alsing 2018 (arxiv 1709.07889)

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    a_dist = lc_utils.alsing_pdf(a=1.1, b=2.8)
    # a_dist = lc_utils.normalized_dist(pdf=lc_utils.alsing_pdf, a=1.1, b=2.8)
    m1 = a_dist.rvs(size=mass_draws)
    m2 = a_dist.rvs(size=mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def BNS_farrow(mass_draws):
    '''
    From Farrow 2019 (arxiv 1902.03300)

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    f_dist = lc_utils.farrow_pdf(a=1.1, b=2.8)
    # f_dist = lc_utils.normalized_dist(pdf=lc_utils.farrow_pdf, a=1.1, b=2.8)
    m1 = f_dist.rvs(size=mass_draws)
    m2 = 1.1*np.ones(mass_draws)+1.7*np.random.rand(mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type


def NSBH_zhu(mass_draws):
    '''
    From Zhu 2021 (arxiv 2011.02717)

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    z_dist = lc_utils.zhu_pdf(a=2.8, b=25)
    # z_dist = lc_utils.normalized_dist(pdf=zhu_pdf, a=2.8, b=25)
    dist_NS_zhu = stats.norm(1.33, scale=.01)
    m1 = z_dist.rvs(size=mass_draws)
    m2 = dist_NS_zhu.rvs(size=mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def NSBH_LRR(mass_draws):
    '''
    Draw a NS and a BH from LIGO Living Reviews
    in Relativity distribution

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    ns_astro_mass_dist = stats.norm(1.33, 0.09)
    bh_astro_mass_dist = stats.pareto(b=1.3)
    m1 = bh_astro_mass_dist.rvs(size=mass_draws)
    m2 = ns_astro_mass_dist.rvs(size=mass_draws)
    merger_type = 'NSBH'
    return m1, m2, merger_type


def BNS_LRR(mass_draws):
    '''
    Draw NS masses from LIGO Living Reviews
    in Relativity distribution

    Parameters
    ----------
    mass_draws: int
        number of masses to draw from distribution

    Returns
    -------
    m1: float
        larger component mass in solar masses
    m2: float
        smaller component mass in solar masses
    merger_type: str
        BNS or NSBH
    '''
    ns_astro_mass_dist = stats.norm(1.33, 0.09)
    m1 = ns_astro_mass_dist.rvs(size=mass_draws)
    m2 = ns_astro_mass_dist.rvs(size=mass_draws)
    merger_type = 'BNS'
    return m1, m2, merger_type
