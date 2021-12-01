import os
from pathlib import Path
import pkgutil

from urllib import request

__version__ = '0.1.2'

PACKAGE_DATA_LINKS = {
    'knn_em_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/main/ligo/em_bright/data/knn_em_classifier.pkl',  # noqa: E501
    'knn_ns_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/main/ligo/em_bright/data/knn_ns_classifier.pkl',  # noqa: E501
    'equil_2H.dat': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/move-data-dir/data/equil_2H.dat'  # noqa: E501
}

EOS_BAYES_FACTORS = {
    'SLy': 1.0,
    'BHF_BBB2': 0.995,
    'KDE0V': 1.075,
    'KDE0V1': 1.079,
    'SKOP': 0.78,
    'H4': 0.074,
    'HQC18': 1.074,
    'SLY2': 1.01,
    'SLY230A': 0.947,
    'SKMP': 0.356,
    'RS': 0.218,
    'SK255': 0.22,
    'SLY9': 0.448,
    'APR4_EPP': 1.06,
    'SKI2': 0.14,
    'SKI4': 0.392,
    'SKI6': 0.337,
    'SK272': 0.202,
    'SKI3': 0.135,
    'SKI5': 0.035,
    'MPA1': 0.309,
    'HQC18': 1.074,
    'MS1B_PP': 0.014,
    'MS1_PP': 0.002
}
"""Taken from Table I, Approx bayes factor column
of 10.1103/PhysRevD.104.083003
"""

BAYES_FACTOR_NORM = sum(EOS_BAYES_FACTORS.values())
"""Sum of bayes factors above"""

EOS_MAX_MASS = {
    'SLy': 2.054,
    'BHF_BBB2': 1.922,
    'KDE0V': 1.96,
    'KDE0V1': 1.969,
    'SKOP': 1.973,
    'H4': 2.031,
    'HQC18': 2.045,
    'SLY2': 2.054,
    'SLY230A': 2.099,
    'SKMP': 2.107,
    'RS': 2.117,
    'SK255': 2.144,
    'SLY9': 2.156,
    'APR4_EPP': 2.159,
    'SKI2': 2.163,
    'SKI4': 2.17,
    'SKI6': 2.19,
    'SK272': 2.232,
    'SKI3': 2.24,
    'SKI5': 2.24,
    'MPA1': 2.469,
    'MS1B_PP': 2.747,
    'MS1_PP': 2.753
}
"""Taken from Table I m_max column of 10.1103/PhysRevD.104.083003"""
_em_bright_loader = pkgutil.get_loader(__name__)
_em_bright_data_dir = Path(_em_bright_loader.path).parents[0] / 'data'

if not os.path.exists(_em_bright_data_dir):
    os.makedirs(_em_bright_data_dir)


def _download_if_not_exists(filename):
    if os.path.exists(os.path.join(_em_bright_data_dir, filename)):
        return
    print(
        f"{filename} not found. "
        f"Downloading {PACKAGE_DATA_LINKS[filename]}..."
    )
    r = request.urlopen(PACKAGE_DATA_LINKS[filename], timeout=60)
    with open(os.path.join(_em_bright_data_dir, filename), 'wb') as f:
        f.write(r.read())


for ff in PACKAGE_DATA_LINKS:
    _download_if_not_exists(ff)
