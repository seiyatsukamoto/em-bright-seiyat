import numpy as np
import h5py
from pathlib import Path
from scipy.stats import rv_continuous
from scipy.integrate import quad


def load_eos_posterior():
    '''
    loads eos posterior draws (https://zenodo.org/record/6502467#.Yoa2EKjMI2z)

    Returns
    -------
    draws: np.array
        equally weighted eos draws from file
    '''
    rel_path = 'data/LCEHL_EOS_posterior_samples_PSR+GW_slice.h5'
    eos_path = Path(__file__).parents[1] / rel_path
    with h5py.File(eos_path, 'r') as f:
        draws = np.array(f['EOS'])
    return draws


def make_distribution(foo):
    class _MakeDistribution(rv_continuous):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.normalize, _ = quad(foo, self.a, self.b)

        def _pdf(self, m):
            return foo(m) / self.normalize
    return _MakeDistribution


@make_distribution
def alsing_pdf(m):
    '''
    From Alsing 2018 (arxiv 1709.07889)

    Parameters
    ----------
    m: float or np.array
        mass(es) at which the PDF is evaluated

    Returns
    -------
    PDF: float or np.array
        PDF value at m
    '''
    mu1, s1, mu2, s2, a = 1.34, 0.07, 1.8, 0.21, 2.12
    PDF1 = a/(s1*np.sqrt(2*np.pi))*np.exp(-((m-mu1)/(np.sqrt(2)*s1))**2)
    PDF2 = a/(s2*np.sqrt(2*np.pi))*np.exp(-((m-mu2)/(np.sqrt(2)*s2))**2)
    PDF = PDF1+PDF2
    return PDF


@make_distribution
def farrow_pdf(m):
    '''
    From Farrow 2019 (arxiv 1902.03300)


    Parameters
    ----------
    m: float or np.array
        mass(es) at which the PDF is evaluated

    Returns
    -------
    PDF: float or np.array
        PDF value at m
    '''
    mu1, s1, mu2, s2, a = 1.34, 0.02, 1.47, 0.15, 0.68
    PDF1 = a/(s1*np.sqrt(2*np.pi))*np.exp(-((m-mu1)/(np.sqrt(2)*s1))**2)
    PDF2 = (1-a)/(s2*np.sqrt(2*np.pi))*np.exp(-((m-mu2)/(np.sqrt(2)*s2))**2)
    PDF = PDF1+PDF2
    return PDF


@make_distribution
def zhu_pdf(m):
    '''
    From Zhu 2021 (arxiv 2011.02717)

    Parameters
    ----------
    m: float or np.array
        mass(es) at which the PDF is evaluated

    Returns
    -------
    PDF: float or np.array
        PDF value at m
    '''
    a1, a2, a3, b1, b2, b3 = .002845, 1.04e11, 799.1, 1.686, 2.1489, .2904
    PDF = 1/(1/(a1*np.exp(b1*m))+1/(a2*np.exp(-b2*m)+a3*np.exp(-b3*m)))
    return PDF


def lambdas_to_lambdatilde(samples):
    q = samples['m1']/samples['m2']
    lambda1, lambda2 = samples['lambda1'], samples['lambda2']
    samples['lambdatilde'] = (16.0/13.0)*(lambda2 + lambda1*(q**5) + 12*lambda1*(q**4) + 12*lambda2*q)/((q+1)**5)
    return samples


def compactness_to_lambdas(c1, c2):
    lambda_coeff = np.array([374839, -1.06499e7, 1.27306e8, -8.14721e8, 2.93183e9, -5.60839e9, 4.44638e9])
    coeff = lambda_coeff[::-1]
    p = np.poly1d(coeff)
    lambda1 = p(c1)
    lambda2 = p(c2)
    return lambda1, lambda2


class NSBHEjectaFitting(object):
    # from https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/joint/conversion.py
    def __init__(self):
        pass

    def chieff2risco(self, chi_eff):

        Z1 = 1.0 + (1.0 - chi_eff ** 2) ** (1.0 / 3) * (
            (1 + chi_eff) ** (1.0 / 3) + (1 - chi_eff) ** (1.0 / 3)
        )
        Z2 = np.sqrt(3.0 * chi_eff ** 2 + Z1 ** 2.0)

        return 3.0 + Z2 - np.sign(chi_eff) * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2))

    def remnant_disk_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_eff,
        a=0.40642158,
        b=0.13885773,
        c=0.25512517,
        d=0.761250847,
    ):

        mass_ratio_invert = mass_1_source / mass_2_source
        symm_mass_ratio = mass_ratio_invert / np.power(1.0 + mass_ratio_invert, 2.0)

        risco = self.chieff2risco(chi_eff)
        bayon_mass_2 = (
            mass_2_source * (1.0 + 0.6 * compactness_2) / (1.0 - 0.5 * compactness_2)
        )

        remant_mass = (
            a * np.power(symm_mass_ratio, -1.0 / 3.0) * (1.0 - 2.0 * compactness_2)
        )
        remant_mass += -b * risco / symm_mass_ratio * compactness_2 + c

        remant_mass = np.maximum(remant_mass, 0.0)

        remant_mass = np.power(remant_mass, 1.0 + d)

        remant_mass *= bayon_mass_2

        return remant_mass

    def dynamic_mass_fitting(
        self,
        mass_1_source,
        mass_2_source,
        compactness_2,
        chi_eff,
        a1=7.11595154e-03,
        a2=1.43636803e-03,
        a4=-2.76202990e-02,
        n1=8.63604211e-01,
        n2=1.68399507,
    ):

        """
        equation (9) in https://arxiv.org/abs/2002.07728
        """

        mass_ratio_invert = mass_1_source / mass_2_source

        risco = self.chieff2risco(chi_eff)
        bayon_mass_2 = (
            mass_2_source * (1.0 + 0.6 * compactness_2) / (1.0 - 0.5 * compactness_2)
        )

        mdyn = (
            a1
            * np.power(mass_ratio_invert, n1)
            * (1.0 - 2.0 * compactness_2)
            / compactness_2
        )
        mdyn += -a2 * np.power(mass_ratio_invert, n2) * risco + a4
        mdyn *= bayon_mass_2

        mdyn = np.maximum(0.0, mdyn)

        return mdyn


class BNSEjectaFitting(object):
    # from https://github.com/nuclear-multimessenger-astronomy/nmma/blob/main/nmma/joint/conversion.py
    def __init__(self):
        pass

    def log10_disk_mass_fitting(
        self,
        total_mass,
        mass_ratio,
        MTOV,
        R16,
        a0=-1.725,
        delta_a=-2.337,
        b0=-0.564,
        delta_b=-0.437,
        c=0.958,
        d=0.057,
        beta=5.879,
        q_trans=0.886,
    ):

        k = -3.606 * MTOV / R16 + 2.38
        threshold_mass = k * MTOV

        xi = 0.5 * np.tanh(beta * (mass_ratio - q_trans))

        a = a0 + delta_a * xi
        b = b0 + delta_b * xi

        log10_mdisk = a * (1 + b * np.tanh((c - total_mass / threshold_mass) / d))
        log10_mdisk = np.maximum(-3.0, log10_mdisk)

        return log10_mdisk

    def log10_dynamic_mass_fitting_CoDiMaMe(
        self,
        mass_1,
        mass_2,
        compactness_1,
        compactness_2,
        a=-0.0719,
        b=0.2116,
        d=-2.42,
        n=-2.905,
    ):
        """
        See https://arxiv.org/pdf/1812.04803.pdf
        """

        log10_mdyn = (
            a * (1 - 2 * compactness_1) * mass_1 / compactness_1
            + b * mass_2 * np.power(mass_1 / mass_2, n)
            + d / 2
        )

        log10_mdyn += (
            a * (1 - 2 * compactness_2) * mass_2 / compactness_2
            + b * mass_1 * np.power(mass_2 / mass_1, n)
            + d / 2
        )

        return log10_mdyn

    def dynamic_mass_fitting_KrFo(
        self,
        mass_1,
        mass_2,
        compactness_1,
        compactness_2,
        a=-9.3335,
        b=114.17,
        c=-337.56,
        n=1.5465,
    ):
        """
        See https://arxiv.org/pdf/2002.07728.pdf
        """

        mdyn = mass_1 * (
            a / compactness_1 + b * np.power(mass_2 / mass_1, n) + c * compactness_1
        )
        mdyn += mass_2 * (
            a / compactness_2 + b * np.power(mass_1 / mass_2, n) + c * compactness_2
        )
        mdyn *= 1e-3

        mdyn = np.maximum(0.0, mdyn)

        return mdyn
