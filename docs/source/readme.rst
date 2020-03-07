Overview
========

Python Laboratory for Finite Element Analysis (pyLabFEA)
--------------------------------------------------------

:Author: Alexander Hartmaier
:Organization: ICAMS, Ruhr University Bochum, Germany
:Contact: alexander.hartmaier@rub.de

Finite Element Analysis (FEA) is a numerical method for studying
mechanical behavior of fluids and solids. The pyLabFEA package introduces a simple version
of FEA for solid mechanics and elastic-plastic materials. pyLabFEA is
fully written in Python. Due to its simplicity, it is well-suited for teaching, and
its flexibility in constitutive modeling of materials makes it a useful 
research tool.

Jupyter notebooks
-----------------

The pyLabFEA package is conveniently used with Jupyter notebooks. 
The available notebooks with tutorials on FEA,
homogenization of elastic and elastic-plastic material behavior, and
constitutive models based on machine learning algorithms are contained in
subfolder ``notebooks``. An overview on the contents of of the notebooks 
is available in the examples of this documentation.

Documentation
-------------
Documentation for pyLabFEA is generated using Sphinx. 
The HTML documentation can be found at pyLabFEA/docs/build/index.html

Dependencies
------------

pyLabFEA requires the following packages as imports:

  - NumPy_ for array handling
  - Scipy_ for numerical solutions
  - scikit-learn_ for machine learning algorithms
  - MatPlotLib_ for graphical output
  - sphinx_ for generating documentation.
  
.. _NumPy: http://numpy.scipy.org
.. _Scipy: https://www.scipy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _MatPlotLib: https://matplotlib.org/
.. _sphinx: http://www.sphinx-doc.org/en/master/

License
-------
The pyLabFEA package comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under the conditions of the GNU General Public License (`GPLv3`_)

.. _GPLv3: http://www.fsf.org/licensing/licenses/gpl.html