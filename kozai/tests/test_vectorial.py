#! /usr/bin/env python

import json
from numpy.testing import assert_allclose

from ..vectorial import TripleVectorial

###
### Object creation tests
###

def test_make_Triple_vector():
  '''Try to create a Triple_vector class.'''
  tv = TripleVectorial()

def test_set_options():
  '''Create a Triple_vector class with a few options.'''
  tv = TripleVectorial(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)

def test_set_octupole():
  tv = TripleVectorial()
  tv.octupole = False

###
### Parameter calculation tests
###

def test_Phi0():
  '''Test calculation of Phi0.  The calculation is done in units of Solar
  masses, years, and AU.'''
  tv = TripleVectorial(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)
  assert_allclose(tv.Phi0, 383313.6993558)

def test_tsec():
  '''Make sure the calculation of the secular timescale is correct.'''
  tv = TripleVectorial(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=3)
  assert_allclose(tv.tsec, 11625553844.775972)

def test_CKL():
  '''Test the CKL calculation.'''
  tv = TripleVectorial(e1=.1, inc=80, g1=45)
  assert_allclose(tv.CKL, -.00212307888)

###
### Representation tests
###

def test_repr():
  '''Make sure that the triple can print its state.'''
  t = TripleVectorial(inc=80, m1=1e-3)
  t.octupole = False
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['m1'], 1e-3)
  assert_allclose(state['inc'], 80)
  assert state['octupole'] == False

def test_save_as_initial():
  '''Try to set a parameter after creating the object and saving it as an
  initial condition.'''
  t = TripleVectorial()
  t.m1 = 1e-3
  t.save_as_initial()
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['m1'], 1e-3)

###
### Integration tests
###

def test_integrate():
  '''See that we can integrate the triple.'''
  tv = TripleVectorial()
  ev = tv.evolve(1e3)
  assert len(ev) > 0

def test_ecc_extrema():
  '''See that we can use the eccmaxima function.'''
  tv = TripleVectorial()
  ex = tv.extrema(1e4)
  assert len(ex) > 0

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  tv = TripleVectorial()
  tv.cputstop = .1
  tv.evolve(large_time)
  assert tv.t < large_time
