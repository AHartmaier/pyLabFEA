[![DOI](https://zenodo.org/badge/245484086.svg)](https://zenodo.org/badge/latestdoi/245484086) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/pyLabFEA.git/master)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

# pyLabFEA

### Python Laboratory for Finite Element Analysis

  - Authors: Alexander Hartmaier, Ronak Shoghi, Jan Schmidt
  - Organization: [ICAMS](http://www.icams.de/content/) / [Ruhr-Universität Bochum](https://www.ruhr-uni-bochum.de/en), Germany 
  - Contact: <alexander.hartmaier@rub.de>

Finite Element Analysis (FEA) is a numerical method for studying
mechanical behavior of fluids and solids. The pyLabFEA package
introduces a lightweight pure-python version of FEA for solid mechanics and elastic-plastic materials. pyLabFEA can import and analyse data sets on mechanical behavior of materials following the modular materials data schema published on [GitHub](https://github.com/Ronakshoghi/MetadataSchema.git) and described in this [article](https://doi.org/10.1002/adem.202401876). Based on such data, machine learning (ML) yield functions can be trained and used as constitutive models in elasto-plastic FEA. Due to
its simplicity, pyLabFEA is well-suited for teaching, and its flexibility in
constitutive modeling of materials makes it a useful research tool for data-oriented constitutive modeling.

## Features

- pyLabFEA offers a lightweight Python Application Programming Interface (API).
- Object oriented methods for generation of finite element models.
- Flexible methods for material definitions, from conventional continuum elasticity and plasticity methods to training of machine learning yield functions, see this [research article](https://doi.org/10.3389/fmats.2022.868248).
- Import of mechanical data based on the [modular material data schema](https://github.com/Ronakshoghi/MetadataSchema.git), see this [research article](https://doi.org/10.1088/2632-2153/ad379e).

## Installation

The preferred method to use pyLabFEA is within [Anaconda](https://www.anaconda.com) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), into which it can be easily installed from [conda-forge](https://conda-forge.org) by

```
$ conda install pylabfea -c conda-forge
```

Generally, it can be installed within any 
Python environment supporting the package installer for python [pip](https://pypi.org/project/pip/) from its latest [PyPi](https://pypi.org/project/pylabfea/) image via pip

```
$ pip install pylabfea
```

Alternatively, the most recent version of the complete repository, including the source code, documentation and examples, can be cloned and installed locally. It is recommended to create a conda environment before installation. This can be done by the following the command line instructions

```
$ git clone https://github.com/AHartmaier/pyLabFEA.git ./pyLabFEA
$ cd pyLabFEA
$ conda env create -f environment.yml
$ conda activate pylabfea
$ (pylabfea) python -m pip install .
```

The correct installation with this method can be tested with

```
$ (pylabfea) pytest tests -v
```

After this, the package can be used within python, e.g. by importing the entire package with

```python
import pylabfea as fea
```


## Documentation

Online documentation for pyLabFEA can be found under [https://ahartmaier.github.io/pyLabFEA/](https://ahartmaier.github.io/pyLabFEA/).

The documentation has been generated using [Sphinx](http://www.sphinx-doc.org/en/master/).

## Examples
A collection of exemplary notebooks and python scripts is available on pyLabFEA'S [GitHub repository](https://github.com/AHartmaier/pyLabFEA.git) with tutorials on linear and non-linear FEA, homogenization of elastic and
elastic-plastic material behavior, and constitutive models based on
machine learning algorithms.

### Jupyter notebooks

pyLabFEA is conveniently used with Jupyter notebooks, which 
are contained in the subfolder 'notebooks' of this repository and can be accessed via `index.ipynb`. An
overview on the contents of the notebooks is also available in the [documentation](https://ahartmaier.github.io/pyLabFEA/examples.html).

The Jupyter notebooks of the pyLabFEA tutorials are directly accessible on [Binder](https://mybinder.org/v2/gh/AHartmaier/pyLabFEA.git/master)

### Python scripts

Python routines contained in the subfolder 'examples' of this repository demonstrate how ML yield funcitons can be trained based on reference materials with significant plastic anisotropy, as Hill or Barlat reference materials, but also for isotropic J2 plasticity. The training data consists of different stress tensors that mark the onset of plastic yielding of the material. It is important that these stress tensors cover the onset of plastic yielding in the full 6-dimensional stress space, including normal and shear stresses. 

The trained ML flow rules can be used in form of a user material (UMAT) for the commercial FEA package Abaqus (Dassault Systems), as described in the README file in the subdirectory 'src/umat'.

## Contributions

Contributions to the pyLabFEA package are highly welcome, either in form of new 
notebooks with application examples or tutorials, or in form of new functionalities 
to the Python code. Furthermore, bug reports or any comments on possible improvements of 
the code or its documentation are greatly appreciated.

## Dependencies

pyLabFEA requires the following packages as imports:

 - [NumPy](http://numpy.scipy.org) for array handling
 - [Scipy](https://www.scipy.org/) for numerical solutions
 - [scikit-learn](https://scikit-learn.org/stable/) for machine learning algorithms
 - [MatPlotLib](https://matplotlib.org/) for graphical output

## Version history

 - v1: 2D finite element solver for pricipal stresses only
 - v2: Introduction of machine learning (ML) yield functions
 - v3: Generalization of finite element solver and training of ML yield functions to full stress tensors
 - v4: Import and analysis of microstructure-sensitive data on mechanical behavior for training and testing of ML yield functions
 - v4.2: Support of strain hardening in machine learning yield functions
 - v4.3: Import of data on mechanical behavior following the [modular material data schema](https://github.com/Ronakshoghi/MetadataSchema.git) as basis for training of machine learning yield functions
 - v4.4: Support of data with crystallographic texture information

## License

The pyLabFEA package comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of examples, notebooks and documentation are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0 DEED](http://creativecommons.org/licenses/by-nc-sa/4.0/))

&copy; 2025 by Authors, ICAMS/Ruhr University Bochum, Germany