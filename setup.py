#!/usr/bin/env python

import os

from setuptools import find_packages, setup

# ADD needed libraries needed for the end user of the package:
# example:
#      requirements = ["numpy", "scipy>=1.0.0", "requests==2.0.1"
# requirements = ['h5py', 'astropy', 'numpy', 'scipy', 'photutils']

with open("requirements_dev.txt") as requirements_file:
    requirements = requirements_file.read()

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read().replace(".. :changelog", "")


doclink = """
Documentation
-------------

The full documentation can be generated with Sphinx"""


PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name="morphofit",
    version="1.0.0",
    description="A Python morphological analysis package of galaxies",
    long_description=readme + "\n\n" + doclink + "\n\n" + history,
    author="Luca Tortorelli",
    author_email="Luca.Tortorelli@physik.lmu.de",
    url="https://github.com/torluca/morphofit",
    download_url="https://github.com/torluca/morphofit/archive/refs/tags/v1.0.0-beta.tar.gz",
    packages=find_packages(include=["morphofit"]),
    include_package_data=True,
    install_requires=requirements,
    license="MIT License",
    zip_safe=False,
    keywords=["morphofit", "morphology", "galfit"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
    ],
)
