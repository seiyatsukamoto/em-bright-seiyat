import json
from concurrent.futures import (
    as_completed,
    ThreadPoolExecutor,
)
from os import getenv
from os.path import basename

from astropy.utils.console import ProgressBar
from astropy.utils.data import (
    download_file,
    is_url_in_cache,
)

__version__ = '1.1.0.dev1'

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


def _download_data_file(url):
    """Download a single file.

    This is an internal method designed to be called as part of a map()
    operation only.
    """
    return download_file(
        url,
        cache=True,
        show_progress=False,
        pkgname="ligo.em_bright",
    )


def _download_data_files(
    urls,
    message=None,
    max_workers=int(getenv("OMP_NUM_THREADS", 0)) or None,
    **kwargs,
):
    """Download a list of data files using a thread pool.
    """
    urls = list(urls)
    # only download files we need (mainly to constrain the progress bar)
    needed = list(filter(
        lambda x: not is_url_in_cache(x, pkgname="ligo.em_bright"),
        urls,
    ))
    if needed:
        if message:
            print(message)
        # use a threadpool to speed things up
        with ProgressBar(len(needed)) as bar, \
             ThreadPoolExecutor(max_workers=max_workers, **kwargs) as executor:
            for i, result in enumerate(as_completed(
                executor.submit(_download_data_file, url)
                for url in needed
            )):
                bar.update(i+1)
    # run it again with _all_ of the files to get the local paths
    return dict(zip(
        map(basename, urls),
        map(_download_data_file, urls)
    ))


PACKAGE_FILENAMES = _download_data_files(
    PACKAGE_DATA_LINKS.values(),
    message="Downloading ligo.em_bright data files...",
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
