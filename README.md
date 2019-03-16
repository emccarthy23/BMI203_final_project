# BMI203_final_project
Final Project

# Final Project Travis Setup

[![Build
Status](https://travis-ci.org/emccarthy23/BMI203_final_project.svg?branch=master)](https://travis-ci.org/emccarthy23/BMI203_final_project)


## assignment

1. Implement a machine learning algorithm to classify RAP1 binding sites


## usage

To use the package, first run

```
conda install --yes --file requirements.txt
```

to install all the dependencies in `requirements.txt`. Then the package's
main function (located in `BMI203_final_project/__main__.py`) can be run as
follows

```
python -m BMI203_final_project -P data test.txt
```

## testing

Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.

