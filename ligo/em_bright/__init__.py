import json

from astropy.utils.data import download_file

__version__ = '1.0.3'

PACKAGE_DATA_BASE_URL = (
    'https://git.ligo.org/emfollow/em-properties/em-bright/'
    f'-/raw/v{__version__}/ligo/em_bright/data'
)
PACKAGE_DATA_LINKS = {name: f'{PACKAGE_DATA_BASE_URL}/{name}' for name in (
    'equil_2H.dat',
    'APR4_EPP.pickle',
    'BHF_BBB2.pickle',
    'H4.pickle',
    'HQC18.pickle',
    'KDE0V1.pickle',
    'KDE0V.pickle',
    'MPA1.pickle',
    'MS1B_PP.pickle',
    'MS1_PP.pickle',
    'RS.pickle',
    'SK255.pickle',
    'SK272.pickle',
    'SKI2.pickle',
    'SKI3.pickle',
    'SKI4.pickle',
    'SKI5.pickle',
    'SKI6.pickle',
    'SKMP.pickle',
    'SKOP.pickle',
    'SLY230A.pickle',
    'SLY2.pickle',
    'SLY9.pickle',
    'SLy.pickle',
    'EOS_BAYES_FACTOR_MAP.json',
    'EOS_MAX_MASS_MAP.json',
)}

PACKAGE_FILENAMES = dict.fromkeys(PACKAGE_DATA_LINKS)

for name, url in PACKAGE_DATA_LINKS.items():
    PACKAGE_FILENAMES[name] = download_file(
        url, cache=True, pkgname='ligo.em_bright'
    )

# load and normalize bayes factors
with open(PACKAGE_FILENAMES['EOS_BAYES_FACTOR_MAP.json']) as f:
    EOS_BAYES_FACTORS = json.load(f)
    """Taken from Table II (broad prior), Approx bayes factor column
    of 10.1103/PhysRevD.104.083003
    """

_BAYES_FACTOR_NORM = sum(EOS_BAYES_FACTORS.values())
"""Sum of bayes factors above"""

EOS_BAYES_FACTORS.update(
    {k: v / _BAYES_FACTOR_NORM for k, v in EOS_BAYES_FACTORS.items()}
)

with open(PACKAGE_FILENAMES['EOS_MAX_MASS_MAP.json']) as f:
    EOS_MAX_MASS = json.load(f)
    """Taken from Table I m_max column of 10.1103/PhysRevD.104.083003"""
