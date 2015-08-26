#! /usr/bin/env python

import json
from numpy.testing import assert_allclose

from kozai.delaunay import TripleDelaunay

###
### Object creation tests
###

def test_make_Triple():
  '''Try to create a Triple class.'''
  t = TripleDelaunay()
  assert t

def test_nooct():
  '''Try to turn the octupole term off.'''
  t = TripleDelaunay()
  t.octupole = False
  assert t.octupole == False

def test_setoptions():
  '''Try to set some options.'''
  t = TripleDelaunay(a1=1, a2=20, e1=.1, e2=.3, m1=1, m2=.5, m3=2, g1=90, 
        g2=25, r1=1, r2=2, inc=70)
  assert_allclose(t.inc, 70)

def test_set_after_creation():
  '''Try to set a parameter after the object has been created.'''
  t = TripleDelaunay(m1=1, m2=1, m3=1, inc=80)
  t.m2 = 1e-3
  assert_allclose(t.inc, 80)

###
### Parameter calculation tests
###

def test_Th():
  '''Test the calculation of Kozai's integral.'''
  t = TripleDelaunay(e1=.05, inc=70)
  assert_allclose(t.Th, .11668533399441)

def test_epsoct():
  '''Test the epsoct calculation.'''
  t = TripleDelaunay(a1=1, a2=20, e2=.3)
  assert_allclose(t.epsoct, .01648351648)

def test_CKL():
  '''Test the CKL calculation.'''
  t = TripleDelaunay(e1=.1, inc=80, g1=45)
  assert_allclose(t.CKL, -.00212307888)

def test_Hhatquad():
  t = TripleDelaunay(e1=.1, inc=80, g1=45)
  assert_allclose(t.Hhatquad, 1.846364030293)

###
### Representation tests
###

def test_repr():
  '''Make sure that the triple can print its state.'''
  t = TripleDelaunay(inc=80, m2=1e-3)
  t.octupole = False
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['m2'], 1e-3)
  assert_allclose(state['inc'], 80)
  assert state['octupole'] == False

def test_save_as_initial():
  '''Try to set a parameter after creating the object and saving it as an
  initial condition.'''
  t = TripleDelaunay()
  t.m2 = 1e-3
  t.save_as_initial()
  j = t.__repr__()
  state = json.loads(j)
  assert_allclose(state['m2'], 1e-3)

###
### Integration tests
###

def test_evolve():
  '''Try to integrate a triple.'''
  t = TripleDelaunay()
  ev = t.evolve(1e3)
  assert len(ev) > 0

def test_extrema():
  '''See that we can use the eccmaxima function.'''
  t = TripleDelaunay()
  ev = t.extrema(1e4)
  assert len(ev) > 0

def test_cputimeout():
  '''Make sure that the integration halts after exceeding the maximum CPU
  integration time.'''
  large_time = 1e9
  t = TripleDelaunay()
  t.cputstop = .1
  t.evolve(large_time)
  assert t.t < large_time

def test_find_flip():
  '''Make sure that flips are being found.'''
  t = TripleDelaunay()
  t.m2 = 1e-3
  flips = t.find_flips(1e5)
  assert len(flips) > 0

def test_reset():
  '''Try to reset the triple.'''
  t = TripleDelaunay()
  tstop = 1e3
  burn = t.evolve(tstop)
  t.reset()
  ev1 = t.evolve(tstop)
  t.reset()
  ev2 = t.evolve(tstop)
  assert_allclose(ev1[-1], ev2[-1], rtol=1e-2, atol=1e-2)

def test_collision():
  '''Make sure that collisions happen (or not).'''
  t = TripleDelaunay(r1=1e-3, r2=1e-3)
  ev = t.evolve(1e4)
  assert not t.collision

  t = TripleDelaunay(r1=30, r2=30)
  ev = t.evolve(1e4)
  assert t.collision

# todo test conservation of constants
