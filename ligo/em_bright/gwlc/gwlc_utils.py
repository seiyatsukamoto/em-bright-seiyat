import numpy as np
from scipy.stats import rv_continuous
from scipy.integrate import quad

def alsing_pdf(m):
        # From Alsing 2018 (arxiv 1709.07889)
        mu1, sig1, mu2, sig2, a = 1.34, 0.07, 1.8, 0.21, 2.12
        PDF1 = a/(sig1*np.sqrt(2*np.pi))*np.exp(-((m-mu1)/(np.sqrt(2)*sig1))**2)
        PDF2 = a/(sig2*np.sqrt(2*np.pi))*np.exp(-((m-mu2)/(np.sqrt(2)*sig2))**2)
        PDF = PDF1+PDF2
        return PDF

def farrow_pdf(m):
        # From Farrow 2019 (arxiv 1902.03300)
        mu1, sig1, mu2, sig2, a = 1.34, 0.02, 1.47, 0.15, 0.68
        PDF1 = a/(sig1*np.sqrt(2*np.pi))*np.exp(-((m-mu1)/(np.sqrt(2)*sig1))**2)
        PDF2 = (1-a)/(sig2*np.sqrt(2*np.pi))*np.exp(-((m-mu2)/(np.sqrt(2)*sig2))**2)
        PDF = PDF1+PDF2
        return PDF

def zhu_pdf(m):
        # From Zhu 2021 (arxiv 2011.02717)
        a1, a2, a3, b1, b2, b3 = .002845, 1.04e11, 799.1, 1.686, 2.1489, .2904
        PDF = 1/(1/(a1*np.exp(b1*m))+1/(a2*np.exp(-b2*m)+a3*np.exp(-b3*m)))
        return PDF

class alsing_dist(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize, _ = quad(alsing_pdf, self.a, self.b)

    def _pdf(self, m):
        return alsing_pdf(m) / self.normalize


class farrow_dist(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize, _ = quad(farrow_pdf, self.a, self.b)

    def _pdf(self, m):
        return farrow_pdf(m) / self.normalize

class zhu_dist(rv_continuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.normalize, _ = quad(zhu_pdf, self.a, self.b)

    def _pdf(self, m):
        return zhu_pdf(m) / self.normalize
