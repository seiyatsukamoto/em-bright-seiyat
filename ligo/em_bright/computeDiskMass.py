import numpy as np
import astropy.units as u
import astropy.constants as c
from scipy.optimize import fsolve
from scipy.interpolate import UnivariateSpline

import lal
import lalsimulation as lalsim

from . import PACKAGE_FILENAMES


def compute_isco(chi_bh):
    '''
    This function takes as input the aligned spin component of the BH and
    returns the innermost stable circular orbit radius normalized by the mass
    of the BH.
    '''
    if not np.all([np.abs(chi_bh) <= 1.0]):
        raise ValueError('|chi1| must be less than 1.0')
    Z1 = 1.0 + ((1.0 - chi_bh**2)**(1./3.))*((1 + chi_bh)**(1./3.) +
                                             (1 - chi_bh)**(1./3.))
    Z2 = np.sqrt(3*chi_bh**2 + Z1**2)
    r_isco = 3 + Z2 - np.sign(chi_bh) * np.sqrt((3 - Z1)*(3 + Z1 + 2*Z2))
    return r_isco


def _lalsim_neutron_star_radius(m, max_mass, fam):
    '''Return neutron star radius in meters'''
    if m > max_mass:
        return m * 1500  # 1 solar mass = 1500m
    else:
        try:
            return lalsim.SimNeutronStarRadius(m * lal.MSUN_SI, fam)
        except RuntimeError:
            # FIXME handle RuntimeError for edge cases
            # Raised if m is close to max_mass
            # GSL function failed: interpolation error (errnum=1)
            # XLAL Error - <GSL function> (interp.c:150): Generic failure
            return m * 1500


def _r_ns_from_lal_simulation(m_ns, eosname):
    eos = lalsim.SimNeutronStarEOSByName(eosname)
    fam = lalsim.CreateSimNeutronStarFamily(eos)
    max_mass = lalsim.SimNeutronStarMaximumMass(fam)/c.M_sun.value
    try:
        iter(m_ns)
        R_ns = np.array([_lalsim_neutron_star_radius(m, max_mass, fam)
                        for m in m_ns])
    except TypeError:
        R_ns = _lalsim_neutron_star_radius(m_ns, max_mass, fam)
    return R_ns, max_mass


def _compactness_baryon_mass(m_ns, r_ns):
    C_ns = c.G * m_ns * c.M_sun/(r_ns * u.m * c.c**2)
    C_ns = C_ns.value

    d1 = 0.619
    d2 = 0.1359
    BE = (d1*C_ns + d2*C_ns*C_ns) * m_ns  # arXiv:1601.06083 (Eq: 21)
    m2_b = m_ns + BE  # Baryonic mass - Gravitational mass = Binding Energy
    return [C_ns, m2_b]


def computeCompactness(M_ns, eosname='2H', R_ns=None,
                       max_mass=2.834648092299807):
    '''
    Return the neutron star compactness as a function of mass
    and equation of state or radius

    Parameters
    ----------
    M_ns : array_like
        Neutron star mass in solar masses
    eosname : str
        Neutron star equation of state to be used
    R_ns : array_like
        Neutron star radius.
    max_mass : float
        Maximum mass of neutron star.

    Returns
    -------
    [C_ns, m2_b, max_mass]
        Compactness, baryon mass and maximum neutron star mass
        in solar masses.

    Notes
    -----
    The radius and maximum mass of the neutron star is
    inferred based on the equation of state supplied.
    Supply explicitly if `eosname=None`.

    Examples
    --------
    >>> computeCompactness(2.8)
    [array(0.298), array(3.354), 2.834]
    >>> computeDiskMass.computeCompactness(2.9, eosname='AP4')
    [0.5, 0.0, 2.212]
    >>> m_ns = np.array([1.1, 1.2, 1.3])
    >>> computeDiskMass.computeCompactness(m_ns, eosname='AP4')
    [array([0.141, 0.154, 0.167]), array([1.199, 1.318, 1.439]), 2.212]
    >>> computeCompactness(2.1, R_ns=16000, eosname=None, max_mass=2.5)
    [0.193, 2.362, 2.5]
    '''
    if R_ns:
        C_ns, m2_b = _compactness_baryon_mass(M_ns, R_ns)
    elif eosname != '2H':
        # infer radius and maximum mass based on lalsimulation EoS
        R_ns, max_mass = _r_ns_from_lal_simulation(M_ns, eosname)
        C_ns, m2_b = _compactness_baryon_mass(M_ns, R_ns)
    else:
        with open(PACKAGE_FILENAMES['equil_2H.dat'], 'rb') as f:
            M_g, M_b, Compactness = np.loadtxt(f, unpack=True)
        max_mass = 2.834648092299807
        s = UnivariateSpline(M_g, Compactness, k=5)
        s_b = UnivariateSpline(M_g, M_b, k=5)
        C_ns = s(M_ns)
        m2_b = s_b(M_ns)
    try:
        C_ns[M_ns > max_mass] = 0.5  # BH compactness set to 0.5
        m2_b[M_ns > max_mass] = 0.0  # BH baryon mass set to 0.0
    except TypeError:  # if C_ns is not an array
        if M_ns > max_mass:
            C_ns = 0.5
            m2_b = 0.0
    return [C_ns, m2_b, max_mass]


'''
For Newtonian: Obtained from arXiv 1807.00011
alpha = 0.406
beta = 0.139
gamma = 0.255
delta = 1.761

For Kerr: Obtained from private message from F. Foucart
alpha = 0.24037857
beta = 0.15184471
gamma = 0.35607169
delta = 1.81491784
'''


def xi_implicit(xi, kappa, chi1, eta):
    xi_implicit = 3*(xi**2 - 2*kappa*xi + kappa**2*chi1**2)
    xi_implicit /= (xi**2 - 3*kappa*xi + 2*chi1*np.sqrt(kappa**3*xi))
    xi_implicit -= eta*xi**3
    return xi_implicit


def computeDiskMass(m1, m2, chi1, chi2, eosname='2H', kerr=False,
                    R_ns=None, max_mass=None):
    """
    This function computes the remnant disk mass after the coalescence using
    the equation (4) arXiv 1807.00011.

    Parameters
    ----------
    m1, m2 : array_like
        primary and secondary mass(es)
    chi1, chi2 : array_like
        primary and secondary spin(s)
    eosname : str
        Name of the equation of state to be used. `AP4`
        `None` when supplying `R_ns`.
    kerr : bool
        Supply to use the relativistic tidal parameter. See Fishbone (1971).
    R_ns : float
        Radius of the secondary, assuming it is a neutron star.
    max_mass : float
        Maximum mass of a neutron star. To be supplied if not supplying EoS.

    Example
    -------
    >>> computeDiskMass(5.0, 2.0, 0.99, 0.)
    0.6321412881595185
    >>> m1 = np.array([5., 6., 7.])
    >>> m2 = np.array([1.0, 1.2, 1.6])
    >>> chi1 = np.zeros(3)
    >>> chi2 = np.zeros(3)
    >>> computeDiskMass(m1, m2, chi1, chi2)
    array([0.12833991, 0.05054819, 0.])
    >>> computeDiskMass(m1, m2, chi1, chi2, eosname='AP4')
    array([0.00851525, 0., 0.])
    >>> max_mass = 3.0  # in m_sun
    >>> r_ns = 15000.  # in meters
    >>> computeDiskMass(m1, m2, chi1, chi2,
    ...                 R_ns=r_ns, max_mass=max_mass, eosname=None)
    array([0.12265712, 0.04272054, 0.])

    Notes
    -----
    The primary mass, by convention, is larger one. If arrays are supplied,
    `m1`, `m2`, `chi1`, `chi1` should be of the same size.
    """
    try:
        _ = iter(m1)
        assert m1.size == m2.size == chi1.size == chi2.size, \
            "masses and spins must have the same dimensions"
        # Making sure that the supplied m1 >= m2 ##
        largerMass1 = m1 >= m2
        largerMass2 = m2 > m1
        m_primary = np.zeros_like(m1)
        m_secondary = np.zeros_like(m2)
        chi_primary = np.zeros_like(chi1)
        chi_secondary = np.zeros_like(chi2)

        m_primary[largerMass1] = m1[largerMass1]
        m_primary[~largerMass1] = m2[largerMass2]
        m_secondary[~largerMass2] = m2[~largerMass2]
        m_secondary[largerMass2] = m1[~largerMass1]

        chi_primary[largerMass1] = chi1[largerMass1]
        chi_primary[~largerMass1] = chi2[~largerMass1]
        chi_secondary[~largerMass2] = chi2[~largerMass2]
        chi_secondary[largerMass2] = chi1[~largerMass1]
    except TypeError:
        if m1 >= m2:
            m_primary = m1
            m_secondary = m2
            chi_primary = chi1
            chi_secondary = chi2
        else:
            m_primary = m2
            m_secondary = m1
            chi_primary = chi2
            chi_secondary = chi1

    [C_ns, m2_b, max_mass] = computeCompactness(m_secondary, eosname=eosname,
                                                R_ns=R_ns, max_mass=max_mass)

    eta = m_primary*m_secondary/(m_primary + m_secondary)**2
    BBH = m_secondary > max_mass
    BNS = m_primary < max_mass
    # FIXME
    if type(BNS) == bool:
        if BNS or BBH:
            return float(not(BBH) or BNS)

    if not kerr:
        alpha = 0.406
        beta = 0.139
        gamma = 0.255
        delta = 1.761
        firstTerm = alpha*(1 - 2*C_ns)/(eta**(1/3))
        secondTerm = beta*compute_isco(chi1)*C_ns/eta
        thirdTerm = gamma
    else:
        alpha = 0.24037857
        beta = 0.15184471
        gamma = 0.35607169
        delta = 1.81491784
        kappa = (m_primary/m_secondary)*C_ns
        try:
            Num = len(kappa)
            xi = fsolve(xi_implicit, 100*np.ones(Num), args=(kappa, chi1, eta))
        except TypeError:
            xi = fsolve(xi_implicit, 100, args=(kappa, chi1, eta))

        firstTerm = alpha*xi*(1 - 2*C_ns)
        secondTerm = beta*compute_isco(chi1)*C_ns/eta
        thirdTerm = gamma
    combinedTerms = firstTerm - secondTerm + thirdTerm

    if hasattr(combinedTerms, 'ndim') and combinedTerms.ndim >= 1:
        lessthanzero = combinedTerms < 0.0
        combinedTerms[lessthanzero] = 0.0
    else:
        combinedTerms = 0 if combinedTerms < 0 else combinedTerms
    M_rem = (combinedTerms**delta)*m2_b
    if type(BNS) == np.ndarray:
        M_rem[BBH] = 0.0  # BBH is always EM-Dark
        M_rem[BNS] = 1.0  # Ad hoc value, assuming BNS is always EM-Bright
    return M_rem
