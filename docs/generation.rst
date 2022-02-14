=======================
Generating the datasets
=======================

We begin with an existing mock Universe. Let's take the `CosmoDC2 extragalactic catalog<https://arxiv.org/abs/1907.06530>`_ as an example. The raw CosmoDC2 files are available on NERSC. In this tutorial, we look into how to process these files to prepare the training set. Processed files that were used in the paper are also stored on NERSC, but one can process any other simulation with any other configuration.


Convergence labels
==================
The `trainval_data` package contains modules to postprocess an existing simulation for inferring convergence. CosmoDC2 contains convergence values computed at around 1' resolution, suitable for weak lensing studies. To add fluctuations at galaxy-galaxy lensing scales of 1", we use the `trainval_data.raytracers.cosmodc2_raytracer` module to perform additional multi-plane raytracing.


Sampling sightlines
===================
We sample sightlines uniformly throughout CosmoDC2. We call them `pointings` throughout the code. Conveniently, the catalog is released in chunks of NSIDE=32 healpixes, so we operate on a healpix-by-healpix basis.


Input photometric catalog
=========================
Once we have the pointings and the convergence labels associated with the pointings, we need to construct the input. This step involves querying galaxies located within an aperture around the pointing and applying some sensible selections, based on survey depth and photometric uncertainties. The selected galaxies comprise our "edgeless graph" input, basically a set. The `trainval_data.graphs.cosmodc2_graph` is responsible for the graph construction.


Usage
=====

On NERSC, run `generate_data_any.py` with the CosmoDC2 healpix ID as the keyword, e.g.

::

$python generate_data_any 10450


The healpix IDs currently available are: 9559, 9686, 9687, 9814, 9815, 9816, 9942, 9943, 10070, 10071, 10072, 10198, 10199, 10200, 10326, 10327, and 10450.
