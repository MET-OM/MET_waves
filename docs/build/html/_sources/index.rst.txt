.. MET_waves documentation master file, created by
   sphinx-quickstart on Sun Nov 14 22:37:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MET_waves's documentation!
=====================================
Tools for data analysis and visualization of MET Norway (https://www.met.no/) wave datasets (e.g., NORA3, WAM4).
The package is under preparation. Some examples are given below:

Plot panarctic map::

   from met_waves import plot_panarctic_map
   plot_panarctic_map(start_time='2020-11-11T14', end_time='2020-11-11T15',product='NORA3', variable='hs', method='timestep')

.. code-block:: rst

.. image:: hs_nora3.png
  :width: 400


Plot time series of a grid point::

   from met_waves import plot_timeseries
   plot_timeseries(start_time='2007-11-08T12', end_time='2007-11-10T23', lon=3.20, lat=56.53, product='NORA3', variable='hs', write_csv=True)

.. code-block:: rst

.. image:: hs_NORA3_ts.png
  :width: 400

Plot 2D spectra of a grid point::

   from met_waves import plot_2D_spectra
   plot_2D_spectra(start_time='2007-11-08T23', end_time='2007-11-10T23', lon=3.20,lat=56.53, product='NORA3')

.. code-block:: rst

.. image:: spec_nora3.png
  :width: 400

Plot TOPAZ data::

   from met_waves import plot_topaz
   plot_topaz(date='1999-02-02', variable='fice')

.. code-block:: rst

.. image:: fice_topaz.png
  :width: 400

.. toctree::
   :maxdepth: 2
   :caption: Contents:



