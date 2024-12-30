Technical guide
===============

Installation
------------

The pyLabFEA package requires an `Anaconda`_ or `Miniconda`_ environment with a recent Python 
version. It can be installed directly from its `GitHub repository`_  with 
the following command

.. _Anaconda: https://www.anaconda.com/products/individual
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _Github repository: https://github.com/AHartmaier/pyLabFEA.git

::

$ python -m pip install git+https://github.com/AHartmaier/pyLabFEA.git

Alternatively, the repository can be cloned locally and installed via

::

$ git clone https://github.com/AHartmaier/pyLabFEA.git
$ cd pyLabFEA.git/trunk/
$ conda env create -f environment.yml
$ conda activate pylabfea$ python -m pip install . --user

The correct implementation can be tested with

::

$ pytest tests -v

After this, the package can by imported into python scripts with

.. code:: python

   import pylabfea as FE
   

Documentation
-------------
Online documentations for pyLabFEA is found 
under https://ahartmaier.github.io/pyLabFEA, for offline use
open pyLabFEA/doc/index.html to browse through the contents. 
The documentation is generated using `Sphinx`_. 

.. _Sphinx: http://www.sphinx-doc.org/en/master/

Contributions
-------------

Contributions to the pyLabFEA package are highly welcome, either in form of new 
notebooks with application examples or tutorials, or in form of new functionalities 
to the Python code. Furthermore, bug reports or any comments on possible improvements of 
the code or its documentation are greatly appreciated.

The latest version of the pyLabFEA package can be found on GitHub: 
https://github.com/AHartmaier/pyLabFEA.git

Dependencies
------------

pyLabFEA requires the following packages as imports:

  - NumPy_ for array handling
  - Scipy_ for numerical solutions
  - scikit-learn_ for machine learning algorithms
  - MatPlotLib_ for graphical output
  - pandas_ for data import
  - fireworks_ direct import of data resulting from fireworks workflows
  
.. _NumPy: http://numpy.scipy.org
.. _Scipy: https://www.scipy.org/
.. _scikit-learn: https://scikit-learn.org/stable/
.. _MatPlotLib: https://matplotlib.org/
.. _pandas: https://pandas.pydata.org/
.. _fireworks: https://materialsproject.github.io/fireworks/

License
-------
The pyLabFEA package comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under the conditions of the GNU General Public License (`GPLv3`_)

.. _GPLv3: http://www.fsf.org/licensing/licenses/gpl.html

The contents of the examples and notebooks are published under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License
(`CC BY-NC-SA 4.0`_)

.. _CC BY-NC-SA 4.0: http://creativecommons.org/licenses/by-nc-sa/4.0/
