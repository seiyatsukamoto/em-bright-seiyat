# Changelog

## [0.1.1] - 2021-10-06

- Make `source_classification_pe` work for both aligned and precessing cases
- Relax h5py>=2.10.0
- Fix documentation

## [0.1.0] - 2021-09-18

- Port over em-bright functionaility from p-astro SHA 0b9c8247
- Switch to poetry for dependency management and packaging
- Demote data directory from a package stage; use pathlib
- Trim DAG writing script to only em-bright training components
- Remove scikit-learn pin; use latest release
- Retrain and replace classifiers using scikit-learn==0.24.2
