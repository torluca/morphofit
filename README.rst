=============================
morphofit
=============================

A Python package for the morphological analysis of galaxies (Tortorelli & Mercurio 2023).

When using this code, please cite Tortorelli et al 2018, 2023, and Peng et al. 2002, 2010.

Author
------

Luca Tortorelli, University Observatory, Ludwig-Maximilians-Universitaet Muenchen, Germany

Main Contributor
----------------
Amata Mercurio, Dipartimento di Fisica "E.R. Caianiello", Università degli Studi di Salerno, Italy

Copyright
---------

Copyright (C) 2019, 2020 Luca Tortorelli, ETH Zurich, Institute for Particle Physics and Astrophysics

Copyright (C) 2021, 2022, 2023 University Observatory, Ludwig-Maximilians-Universitaet Muenchen

Features
--------

* Creation of multiband catalogue with Source Extractor.

* Creation of PSF images using different methodologies.

* GALFIT run on galaxy stamps from the parent image.

* GALFIT run on galaxy regions from the parent image.

* GALFIT run on galaxies from the full image.

Installation
------------

Install via `pip`:

    pip install morphofit

How to use it
-------------

morphofit consists of a series of modules that are run via command line using the esub-epipe Python package (Zuercher et al. 2021,2022).
See the Jupyter notebook `demo_usage.ipynb` in the morphofit/examples folder for more details.

Acknowledgements
--------

The author is grateful to Chien Y. Peng for the development of GALFIT.