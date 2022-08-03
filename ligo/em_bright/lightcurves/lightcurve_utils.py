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
