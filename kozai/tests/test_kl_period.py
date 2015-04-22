#! /usr/bin/env python

from numpy.testing import assert_allclose
from ..delaunay import TripleDelaunay
from ..vectorial import TripleVectorial
from ..kl_period import *

def test_P_out():
  '''Test the P_out calculation.'''
  tv = TripleVectorial(a2=20, m1=1, m3=1)
  assert_allclose(P_out(tv), 63.2455532)

def test_P_in():
  '''Test the P_in calculation.'''
  tv = TripleVectorial(a1=1, m1=1)
  assert_allclose(P_in(tv), 1)

def test_klperiod_oom():
  '''Test the OOM period of KL oscillations.'''
  tv = TripleVectorial(a1=1, a2=20, m1=1, m3=1, e2=.3)
  assert_allclose(kl_period_oom(tv), 1178.965050)

def test_islibrating():
  '''Test the islibrating calculation.'''
  tv = TripleVectorial(e1=.1, inc=80, g1=45)
  assert is_librating(tv) == True
  tv = TripleVectorial(e1=.1, inc=80, g1=0)
  assert is_librating(tv) == False

def test_kl_period():
  '''Test the KL period semi-analytic calculation.'''
  tv = TripleVectorial(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=1,
        g1=0, Omega=np.pi)
  assert_allclose(kl_period(tv), 4195.8240184679735)

def test_numerical_kl_period_vectorial():
  '''See if the numerical calculation of the KL period by explicitly
  integrating the equations of motion works.'''

  # Triple_vector test
  tv = TripleVectorial(a1=1, a2=20, e1=.1, e2=.3, m1=1, m3=1, g1=0,
    Omega=180)
  tv.octupole = False
  assert_allclose(numerical_kl_period(tv, nperiods=3), 4368.4247305281269)

def test_numerical_kl_period_delaunay():
  '''See if the numerical calculation of the KL period by explicitly
  integrating the equations of motion works.'''

  # Triple test
  t = TripleDelaunay(a1=1, a2=20, e1=.1, e2=.3, m1=1, m2=1, m3=1, g1=0, 
    g2=0)
  t.octupole = False
  assert_allclose(numerical_kl_period(t, nperiods=3), 5862.9998908353418)
