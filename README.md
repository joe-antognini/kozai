# Kozai

Kozai is a Python package to evolve hierarchical triples in the secular
approximation.  Hierarchical triples may be evolved either using the
Delaunay formalism (Naoz et al. 2013b) or the vectorial formalism (Katz et
al. 2011).  The quadrupole, octupole, and hexadecapole terms of the
Hamiltonian may be toggled to be included (or not) in the equations of
motion.  Post-Newtonian terms may also be toggled to include both
relativistic precession (PN 1) and gravitational radiation (PN 2.5) using
terms from Blaes et al. (2002). 

The package provides a TripleDelaunay object which may be integrated using
the Delaunay orbital elements and a TripleVectorial which may be integrated
using the eccentricity and angular momentum vectors.  This allows the
integration to occur within the context of an external Python program.

The underlying integrator is from the SciPy ODE package.  By default this
package uses VODE as its integration algorithm, but the algorithm may be
changed to any of the other integration algorithms supported by the SciPy
ODE package.

## Dependencies

-  NumPy
-  SciPy

## References

- Blaes, O., Lee, M.H., & Socrates, A., 2002, ApJ, 578, 775
- Katz, B., Dong, S., & Malhotra, R., 2011, PhRvL, 107, 181101
- Naoz, S., Farr, W.M., Lithwick, Y., Rasio, F.A., & Teyssandier, J., 2013b,
  MNRAS, 431, 2155
