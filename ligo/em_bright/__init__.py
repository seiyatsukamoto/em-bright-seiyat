from astropy.utils.data import download_file

__version__ = '0.1.5'

PACKAGE_DATA_LINKS = {
    'knn_em_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.5/ligo/em_bright/data/knn_em_classifier.pkl',  # noqa: E501
    'knn_ns_classifier.pkl': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.5/ligo/em_bright/data/knn_ns_classifier.pkl',  # noqa: E501
    'equil_2H.dat': 'https://git.ligo.org/deep.chatterjee/em-bright/-/raw/v0.1.5/ligo/em_bright/data/equil_2H.dat'  # noqa: E501
}
PACKAGE_FILENAMES = dict.fromkeys(PACKAGE_DATA_LINKS)

for name, url in PACKAGE_DATA_LINKS.items():
    PACKAGE_FILENAMES[name] = download_file(
        url, cache=True, pkgname='ligo.em_bright'
    )
