.. em-bright documentation master file, created by
   sphinx-quickstart on Thu Sep  2 16:16:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation of EM-Bright
==========================

Introduction:
-------------
Binary system of neutron stars and black holes are some of the strongest
and best understood emitters of gravitational waves. Additionally, when 
a system has a neutron star then there is also a finite probability of 
disrupted baryonic matter present after the coalescence. This disrupted 
matter could be either from physical collision of the two compact objects
as in the case of a binary neutron star (BNS), or from tidal interactions
between the two compact objects with at least one of them being a neutron
star (NS). This baryonic matter is generally extremely neutron rich, and
will undergo r-process neucleosynthesis producing heavy elements that 
subsequently goes through nuclear fission, producing large quantity of
energy. This may result in an electromagnetic signal, commonly known as
a Kilonova. Simultaneous observation of Kilonova and gravitational wave
resulting from the coalescence of the gravitational wave is one of the 
most sought after astrophysical transient phenomenon. However, these
events are much rapidly evolving than supernova, and the localization
region of gravitational wave can be seperal tens of square degree large.
Thus, early knowledge about the possibility of an electromagnetic 
counterpart of a gravitational wave event can be helpful for observers.
The EM-Bright package provides tools for computing probability of the 
presence of electromagnetic counterpart in a merger of two compact binary 
objects.

For a given compact binary coalescence event the EM-Bright code provides
two probabilities, `HasNS`, and `HasRemnant`. The first quantity gives
the probability that there is a neutron star in the binary, where any 
compact object with mass less than 3.0 solar mass is considered a neutron
star. The second quantity is the probability of non-zero tidally disrupted
matter to be present outside the final object after coalescence. To compute
this quantity we use a fitting formula of numerical relativity results as
provided in Foucart_


EM-Bright Calculation:
----------------------
The knowledge of the masses and spins of the binary will allow us to compute the `HasNS`
and `HasRemnant` probabilities. However, the source parameter information are poorly
known in the low-latency, it might be hours before we get the first results from rapid
paremeter estimation to compute directly the EM-Bright probabilities. To address
this issue, we implement a supervised learning technique to compute `HasNS` and 
`HasRemnant` EMBright-paper_. In its current implementation we apply a nearest neighbor
supervised learning technique to train the classifier based on a large set of simulations.
In this study we inject compact binary coalescence signals in noise stream of LIGO and 
Virgo detectors. We recover these injections using the detection pipelines used by the
LIGO/Virgo collaboration. The recovered parameters exhibit deviation from the injected
parameters due to pipeline systematics. We train the classifier to identify the "true"
EM-Bright nature of the event based on the injected parameters where the feature set
is the recovered parameters. 



.. _Foucart: https://arxiv.org/abs/1807.00011
.. _EMBright-paper: https://arxiv.org/abs/1911.00116


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   em_bright
   compute_disk_mass


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
