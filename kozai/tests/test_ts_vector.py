#! /usr/bin/env python

import os
from numpy.testing import assert_allclose
from ..ts_vector import Triple_vector

###
### Object creation tests
###

def test_make_Triple_vector():
  '''Try to create a Triple_vector class.'''
  tv = Triple_vector()

def test_nooct():
  '''Try to turn the octupole term off.'''
  tv = Triple_vector(octupole=False)

def test_set_options():
  '''Create a Triple_vector class with a few options.'''
  tv = Triple_vector(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)

###
### Parameter calculation tests
###

def test_Phi0():
  '''Test calculation of Phi0.  The calculation is done in units of Solar
  masses, years, and AU.'''
  tv = Triple_vector(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)
  assert_allclose(tv.Phi0, .01705410435)

def test_tsec():
  '''Make sure the calculation of the secular timescale is correct.'''
  tv = Triple_vector(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)
  assert_allclose(tv.tsec, 368.4265781)

def test_CKL():
  '''Test the CKL calculation.'''
  tv = Triple_vector(e1=.1, inc=80, argperi=45)
  assert_allclose(tv.CKL, -.00212307888)

###
### Integration tests
###

def test_integrate():
  '''See that we can integrate the triple.'''
  tv = Triple_vector(tstop=10)
  tv.integrate()

def test_integrate_tofile():
  '''See if we can integrate the triple and write to file.'''
  tv = Triple_vector(tstop=10, outfilename='foo.dat')
  tv.integrate()
  os.remove('foo.dat') # Clean up

def test_ecc_extrema():
  '''See that we can use the eccmaxima function.'''
  tv = Triple_vector(tstop=10)
  tv.ecc_extrema()

def test_ecc_extrema_tofile():
  '''See that we can use the eccmaxima function and write to file.'''
  tv = Triple_vector(tstop=10, outfilename='foo.dat')
  tv.ecc_extrema()
  os.remove('foo.dat') # Clean up

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  tv = Triple_vector(tstop=large_time, cputstop=.1)
  tv.integrate()
  assert tv.t < large_time
