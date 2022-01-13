import os
from pathlib import Path
import pkgutil

from urllib import request

__version__ = '0.1.4'

PACKAGE_DATA_LINKS = {
    'knn_em_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.4/ligo/em_bright/data/knn_em_classifier.pkl',  # noqa: E501
    'knn_ns_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.4/ligo/em_bright/data/knn_ns_classifier.pkl',  # noqa: E501
    'equil_2H.dat': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.4/ligo/em_bright/data/equil_2H.dat'  # noqa: E501
}

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
