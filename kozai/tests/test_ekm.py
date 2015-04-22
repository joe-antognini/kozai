#! /usr/bin/env python

import json
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
  to = TripleOctupole()

def test_set_options_delaunay():
  '''See if we can set some options.'''
  to = TripleOctupole(e1=.05, inc=85, Omega=90, g1=25)

def test_set_options_constants():
  '''Set the integrals of motion'''
  to = TripleOctupole(phiq=.15, chi=F(.15), epsoct=.01)

###
### Representation tests
###

def test_repr():
  '''Make sure that the triple can print its state.'''
  t = TripleOctupole(inc=70, Omega=170)
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['inc'], 70)
  assert_allclose(state['Omega'], 170)

def test_save_as_initial():
  '''Try to set a parameter after creating the object and saving it as an
  initial condition.'''
  t = TripleOctupole()
  t.phiq = .015
  t.save_as_initial()
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['phiq'], .015)

###
### Integration tests
###

def test_evolve():
  '''Integrate the object.'''
  to = TripleOctupole()
  to.evolve(10)

def test_const_evolve():
  '''Integrate an object that has been set with the integrals of motion'''
  to = TripleOctupole(phiq=.15, chi=F(.15), epsoct=.01)
  to.evolve(10)

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  to = TripleOctupole()
  to.cputstop = .1
  to.evolve(large_time)
  assert to.t < large_time

def test_reset():
  '''Try to reset the triple.'''
  t = TripleOctupole()
  tstop = 10
  burn = t.evolve(tstop)
  t.reset()
  ev1 = t.evolve(tstop)
  t.reset()
  ev2 = t.evolve(tstop)
  assert_allclose(ev1[-1], ev2[-1], rtol=1e-2, atol=1e-2)

def test_flip_period():
  '''Test the analytic flip period calculation'''
  t = TripleOctupole(e1=.1, inc=80, Omega=180, g1=0, epsoct=1e-2)
  p = t.flip_period()
  assert_allclose(p, 123.6079642436734, atol=1e-3, rtol=1e-3)

def test_numeric_flip_period():
  '''Test the numeric calculation of the flip period'''
  t = TripleOctupole(e1=.1, inc=80, Omega=180, g1=0, epsoct=1e-2)
  p = t.numeric_flip_period(n_flips=3) 
  assert_allclose(p, 123.58242387832222, atol=1e-3, rtol=1e-3)
