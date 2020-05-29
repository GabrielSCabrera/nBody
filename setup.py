"""Setup for the nbody package."""

import setuptools

dependencies = ['numpy', 'matplotlib', 'ffmpeg', 'vpython']
packages = ['nbody', 'nbody.core', 'nbody.lib', 'nbody.utils', 'nbody.config']

with open('README.md', encoding='utf-8') as f:
    README = f.read()

setuptools.setup(
    author = 'Gabriel S. Cabrera',
    author_email = 'gabric@uio.no',
    name = 'nbody',
    license = 'GPLv3',
    description = 'GPU-accelerated N-Body particle simulator with visualizer',
    version = 'v0.0.4',
    long_description = README,
    url = 'https://github.com/GabrielSCabrera/nBody',
    packages = packages,
    python_requires = '>=3.6.8',
    install_requires = dependencies,
    classifiers = [
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education'
    ],
)
