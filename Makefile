.PHONY: help clean clean-pyc clean-build list test test-all coverage docs release sdist

help:
	@echo
	@echo "style-check - check code with flake8"
	@echo "reformat    - runs black to reformat code"
	@echo "test        - run tests quickly with the default Python"
	@echo "test-all    - run tests on every Python version with tox"
	@echo "coverage    - check code coverage quickly with the default Python"
	@echo "docs        - generate Sphinx HTML documentation, including API docs"
	@echo "sdist       - package"

clean: clean-build clean-pyc

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr *.egg-info

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -rf {} +

reformat:
	black -l 100 morphofit tests

style-check:
	tox -e style

test:
	py.test tests

test-all:
	tox

coverage:
	coverage run --source morphofit -m pytest tests
	coverage report -m
	coverage html
	open htmlcov/index.html

docs:
	rm -f docs/morphofit.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ morphofit
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	coverage run --source morphofit -m pytest tests
	coverage report -m
	coverage html
	cp -R htmlcov docs/_build/html
	open docs/_build/html/index.html

sdist: clean
	python setup.py sdist
	ls -l dist
