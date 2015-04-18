#! /usr/bin/env python

import os
from numpy.testing import assert_allclose

from ..ekm import *

###
### Function tests
###

def test_F():
  '''Test the calculation of F.'''
  assert_allclose(F(.015), .017068821850895335)

###
### Object creation tests
###

def test_triple_octupole():
  '''Try to create a Triple_octupole class.'''
  to = Triple_octupole()

def test_set_options():
  '''See if we can set some options.'''
  to = Triple_octupole(a1=2, a2=10, e1=.05, e2=.4, inc=85, longascnode=90,
         argperi=25)
  to = Triple_octupole(phiq=.15, chi=F(.15), epsoct=.01)

###
### Integration tests
###

def test_integrate():
  '''Integrate the object.'''
  to = Triple_octupole(tstop=10)
  to.integrate()

def test_integrate_tofile():
  '''See if we can integrate the triple and write to file.'''
  to = Triple_octupole(tstop=10, outfilename='foo.dat')
  to.integrate()
  os.remove('foo.dat') # Clean up

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  to = Triple_octupole(tstop=large_time, cputstop=.1)
  to.integrate()
  assert to.t < large_time
