__version__ = '1.1.5dev'

PACKAGE_DATA_BASE_URL = (
    'https://git.ligo.org/emfollow/em-properties/em-bright/'
    '-/raw/main/ligo/em_bright/data'
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
    'EOS_POSTERIOR_DRAWS.h5',
    'MASS_GAP.pickle'
)}
