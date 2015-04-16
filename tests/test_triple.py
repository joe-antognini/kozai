#! /usr/bin/env python

import os
from numpy.testing import assert_allclose

from ..triplesec import Triple


###
### Object creation tests
###

def test_make_Triple():
  '''Try to create a Triple class.'''
  t = Triple()

def test_nooct():
  '''Try to turn the octupole term off.'''
  t = Triple(octupole=False)

def test_setoptions():
  '''Try to set some options.'''
  t = Triple(a1=1, a2=20, e1=.1, e2=.3, m1=1, m2=.5, m3=2, argperi1=90,
        argperi2=25, r1=1, r2=2, inc=70)

###
### Integration tests
###

def test_integrate():
  '''Try to integrate a triple.'''
  t = Triple(tstop=10)
  t.integrate()

def test_integrate_tofile():
  '''See if we can integrate the triple and write to file.'''
  t = Triple(tstop=10, outfilename='foo.dat')
  t.integrate()
  os.remove('foo.dat') # Clean up

def test_ecc_extrema():
  '''See that we can use the eccmaxima function.'''
  t = Triple(tstop=10)
  t.ecc_extrema()

def test_ecc_extrema_tofile():
  '''See that we can use the eccmaxima function and write to file.'''
  t = Triple(tstop=10, outfilename='foo.dat')
  t.ecc_extrema()
  os.remove('foo.dat') # Clean up

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  t = Triple(tstop=large_time, cputstop=.1)
  t.integrate()
  assert t.t < large_time

# todo test conservation of constants
