# bird_diversity

An example repo for a bird diversity experiment.

This example is based on a classic ecology paper that uses Shannon Entropy
(also called Shannon Index in ecology) as a measure of diversity of bird and
plant species and of foliage cover at different heights to address the question:
"What is it about the environment which controls the bird species diversity?"

The original paper is:
MacArthur, R.H. and MacArthur, J.W. (1961), On Bird Species Diversity. Ecology, 42: 594-598. https://doi.org/10.2307/1932254

In our example, we are interested in replicating this experiment, so we have
gathered data from a number of test sites across the US.

## git hash based version number

This package uses setuptools_scm to set the version number of the package equal
based on the git hash. To get the version number (which contains the hash) use:
```
import bird_diversity
version = bird_diversity.__version__
```

Then the `version` variable will contain a version string, something like:
```
'0.1.dev2+g81f125f92.d20260204'
```
The part after the `+g` is the git hash (N.B. the "g" is not part of the hash).
If there is anything after the hash (e.g. `.d20260204` in this example) it means
that there are local changes that are not included in a commit.
There will be no `+g` if you are on a tagged version because the tag fully
specifies the commit. 


## testing

To run tests, you must first create a conda environment with the required
packages Use:
`conda env create -f environment.yaml`

Next install the bird_diversity package with pip. Use:
`pip install --no-deps -e .`

(The option `--no-deps` tells pip not to install the dependencies, something we want to do because we manage our dependencies with conda, not pip. The option `-e` makes this an editable install so that if you change you don't have to reinstall to run the modified code.)

You can run the tests in the `tests` directory with:
`pytest`

A few things to note:
1. It is important to have an installable package, with an `__init__.py` file in the package directory and a `pyproject.toml`.
2. You must install the package to be able to run the tests.
3. Both the test files and the test functions must start with `test_` for pytest to find them.
4. In the test files do not use relative imports.

## continuous integration

We are using GitHub actions for our continuous integration. GitHub actions looks
for workflow files, which are yaml files that specify what to run on the GitHub
action runner. The workflow files **must** be placed in the `.github/workflows/`
folder and must have a `.yaml` or `.yml` extension for GitHub actions to find them.

In our action, we first set up our conda environment using the `environment.yaml`
file in our repo and then call `pytest` to run our tests. We also use the pytest-cov
plugin to report our test coverage.

Note:
GitHub actions are disabled by default on forks.