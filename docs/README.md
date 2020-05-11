![Build Status](https://gitlab.com/suprcongpu/suprcongpu/badges/master/pipeline.svg)

---

## GitLab CI

This project's static Pages are built by GitLab CI, following the steps
defined in [`.gitlab-ci.yml`](.gitlab-ci.yml):

```
image: python:3-alpine

pages:
  stage: deploy
  script:
    - pip install -r docs/requirements.txt
    - sphinx-build -b html docs/source public
  artifacts:
    paths:
      - public
  only:
    - master
```

## Requirements
The requirements are handled explicitly by `pip` via [`requirements.txt`](docs/requirements.txt)

- [Sphinx](https://www.sphinx-doc.org)
- [Read The Docs Sphinx Theme](https://sphinx-rtd-theme.readthedocs.io/en/stable/)

## How to use locally
In order to run and preview locally, clone the repo. To compile and see the changes, follow the steps below (you need to create a virtual environment only once).

Create a Python 3 virtual environment using the venv module included with Python 3 (outside the cloned repository)
```
python -m venv py3-sphinx
```
Now “activate” the environment. Look for the name of the virtual environment enclosed in parenthesis after activation
```
source py3-sphinx/bin/activate
```
Go into the cloned repository folder and install the necessary requirements
```
pip install -r docs/requirements.txt
```
In order to preview changes, go into the `docs/` folder and run (inside `docs/`)
```
make html
```
This creates a new directory structure `public/html/` which contains the root of the documentation. Open it with the browser of your choice, e.g.
```
firefox public/html/index.html &
```
Upon adding new changes and rebuilding the docs, refresh the browser tab to see the changes.
