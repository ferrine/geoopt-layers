geoopt layers
=============

Experimental layers to be used along with `geoopt`_ package

Design
------
With recent `geoopt`_ Manifold tensors have become first class citizens in pytorch autograd.
It is extremely convenient to use the supplied manifold information about the tensor.
This should make Riemannian deep learning closer to the community.
However, we hound it useful to enforce some constraints in the design.

- All input tensors should be Manifold tensors with same manifold attached as in the layer.
- All output tensors should have manifold attached.

These two rules make initialization a bit cumbersome, but overall robustness to bugs increases.

.. _geoopt: <https://github.com/geoopt/geoopt>