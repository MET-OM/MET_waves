#!/usr/bin/env python

import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))
exec(open(os.path.join(here, 'version.py')).read())

setuptools.setup(
    name        = 'MET_waves',
    description = 'MET_waves - Tools for data analysis and visualization of MET Norway (https://www.met.no/) wave datasets (e.g., NORA3, WAM4)',
    author      = 'Konstantinos Christakos NTNU & MET Norway',
    url         = 'https://github.com/KonstantinChri/MET_waves',
    download_url = 'https://github.com/KonstantinChri/MET_waves',
    version = __version__,
    license = 'MIT',
    install_requires = [
        'numpy',
        'scipy',
        'matplotlib',
        'netCDF4'
    ],
    packages = setuptools.find_packages(),
    include_package_data = True,
    setup_requires = ['setuptools_scm'],
    tests_require = ['pytest'],
    scripts = []
)
