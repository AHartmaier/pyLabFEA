# pyLabFEA

### Python Laboratory for Finite Element Analysis

  - Author: Alexander Hartmaier
  - Organization: ICAMS, Ruhr University Bochum, Germany
  - Contact: <alexander.hartmaier@rub.de>

Finite Element Analysis (FEA) is a numerical method for studying
mechanical behavior of fluids and solids. The pyLabFEA package
introduces a simple version of FEA for solid mechanics and
elastic-plastic materials, which is fully written in Python. Due to
its simplicity, it is well-suited for teaching, and its flexibility in
constitutive modeling of materials makes it a useful research tool.

## Installation

The pyLabFEA package can be installed directly from the GitHub repository with the following command

```
$ python -m pip install git+https://github.com/AHartmaier/pyLabFEA.git
```

Alternatively, the repository can be cloned locally and installed via

```
$ python -m pip install . --user
```

from within the `trunk` directory. After this, the package can by imported into python with

```python
import pylabfea as FE
```

The correct implementation can be tested with

```
$ pytest tests
```

## Documentation

Online documentation for pyLabFEA can be found under https://ahartmaier.github.io/pyLabFEA/.
For offline use, open pyLabFEA/docs/index.html to browse through the contents.
The documentation is generated using [Sphinx](http://www.sphinx-doc.org/en/master/).

## Jupyter notebooks

pyLabFEA is conveniently used with Jupyter notebooks. 
Available notebooks with tutorials on linear and non-linear FEA, homogenization of elastic and
elastic-plastic material behavior, and constitutive models based on
machine learning algorithms are contained in subfolder `notebooks`. An
overview on the contents of the notebooks is available [here](https://ahartmaier.github.io/pyLabFEA/examples.html).

The Jupyter notebooks of the pyLabFEA tutorials are also available on Binder 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/AHartmaier/pyLabFEA.git/master)


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
 - [pandas](https://pandas.pydata.org/) for data import

## License

The pyLabFEA package comes with ABSOLUTELY NO WARRANTY. This is free
software, and you are welcome to redistribute it under the conditions of
the GNU General Public License
([GPLv3](http://www.fsf.org/licensing/licenses/gpl.html))

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
([CC BY-NC-SA 4.0](http://creativecommons.org/licenses/by-nc-sa/4.0/))
