#! /usr/bin/env python 

#
# secular
#
# Numerically integrates Equations 11 -- 17 in Blaes, Lee, & Socrates
# (2002).  Now includes the hexadecapole term.
#

from math import sqrt, cos, sin, pi, acos
from numpy import linspace
import pygsl
import pygsl._numobj as Numeric
import time
from pygsl import odeiv
import random

#####  constants  (made to match Fewbody, though in mks) ####
########


def calc_H(e, i, consts, m, a):
  '''Calculates H.  See eq. 22 of Blaes et al. (2002)'''

  a1, a2 = a
  e1, e2 = e
  G, c = consts
  m0, m1, m2 = m

  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))

  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  return sqrt(2 * cos(i) * G1 * G2 + G1**2 + G2**2)

def calc_cosi(consts, m, y):
  '''Calculates the cosine of the inclination given H.'''

  G = consts[0]

  a1 = y[0]
  e1 = y[2]
  a2 = y[3]
  e2 = y[5]
  H = y[6]

  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))
  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  return (H**2 - G1**2 - G2**2) / (2 * G1 * G2)

# TODO: Why is this something here? 
# TODO: Remove input_oct
def deriv(t, y, something):
  '''Compute derivatives of y at t'''
  a1, g1, e1, a2, g2, e2, H = y
  G, c = consts
  m0, m1, m2 = m

  sing1 = sin(g1)
  sing2 = sin(g2)
  cosg1 = cos(g1)
  cosg2 = cos(g2)

  C2 = (G * m0 * m1 * m2 / (16 * (m0 + m1) * a2 * sqrt((1 - e2**2)**3)) *
    (a1 / a2)**2)

  C3 = 0.0
  if input_oct == 1:
      C3 = (15 * G * m0 * m1 * m2 * (m0 - m1) / (64 * (m0 + m1)**2 * a2 * 
        sqrt((1 - e2**2)**5)) * (a1 / a2)**3)
  
  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))

  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  th = (H**2 - G1**2 - G2**2) / (2 * G1 * G2)
  cosphi = - cosg1 * cosg2 - th * sing1 * sing2

  B = 2 + 5 * e1**2 - 7 * e1**2 * cos(2 * g1)

  A = 4 + 3 * e1**2 - 5 / 2. * (1 - th**2) * B


  
  da1dt = 0.0
  if input_gr:
    da1dt += - (64 * G**3 * m0 * m1 * (m0 + m1) / (5 * c**5 * a1**3 * 
      sqrt((1 - e1**2)**7)) * (1 + 73 / 24. * e1**2 + 37 / 96. * e1**4))

  dg1dt = (6 * C2 * (1 / G1 * (4 * th**2 + (5 * cos(2 * g1) - 1) * (1 -
    e1**2 - th**2)) + th / G2 * (2 + e1**2 * (3 - 5 * cos(2 * g1)))) + C3 *
    e2 * e1 * (1 / G2 + th / G1) * (sing1 * sing2 * (A + 10 * (3 *
    th**2 - 1) * (1 - e1**2)) - 5 * th * B * cosphi) - C3 * e2 * (1 -
    e1**2) / (e1 * G1) * (10 * th * (1 - th**2) * (1 - 3 * e1**2) * sing1
    * sing2 + cosphi * (3 * A - 10 * th**2 + 2)))
  if input_gr:
    dg1dt += (3 / (c*c * a1 * (1 - e1**2)) * sqrt((G * (m0 + m1) / a1)**3))

  if input_hex:
    dg1dt += (1 / (4096. * a2**5 * sqrt(1 - e1**2) * (m0 + m1)**5) * 45 *
      a1**3 * sqrt(a1 * G * (m0 + m1)) * (-1 / ((e2**2 - 1)**4 * sqrt(a2 *
      G * (m0 + m1 +m2))) * (m0**2 - m0 * m1 + m1**2) * (sqrt(1 - e2**2) *
      m1**2 * m2 * sqrt(a2 * G * (m0 + m1 + m2)) * th + m0**2 * (sqrt( 1 -
      e1**2) * m1 * sqrt(a1 * G * (m0 + m1)) + sqrt(1 - e2**2) * m2 *
      sqrt(a2 * G * (m0 + m1 + m2)) * th) + m0 * m1 * (sqrt(1 - e1**2) * m1
      * sqrt(a1 * G * (m0 + m1)) + sqrt(1 - e1**2) * sqrt(a1 * G * (m0
      + m1)) * m2 + 2 * sqrt(1 - e2**2) * m2 * sqrt(a2 * G * (m0 + m1 +
      m2)) * th)) * (96 * th + 480 * e1**2 * th + 180 * e1**4 * th + 144 *
      e2**2 * th + 720 * e1**2 * e2**2 * th + 270 * e1**4 * e2**2 * th -
      224 * th**3 - 1120 * e1**2 * th**3 - 420 * e1**4 * th**3 - 336 *
      e2**2 * th**3 - 1680 * e1**2 * e2**2 * th**3 - 630 * e1**4 * e2**2 *
      th**3 + 56 * e1**2 * (2 + e1**2) * (2 + 3 * e2**2) * th * (7 * th**2
      - 4) * cos(2 * g1) - 294 * e1**4 * (2 + 3 * e2**2) * th * (th**2 - 1)
      * cos(4 * g1) - 147 * e1**4 * e2**2 * cos(4 * g1 - 2 * g2) + 441 *
      e1**4 * e2**2 * th**2 * cos(4 * g1 - 2 * g2) + 294 * e1**4 * e2**2 *
      th**3 * cos(4 * g1 - 2 * g2) + 140 * e1**2 * e2**2 * cos(2 * (g1 -
      g2)) + 70 * e1**4 * e2**2 * cos(2 * (g1 - g2)) + 336 * e1**2 * e2**3
      * th * cos(2 * (g1 - g2)) + 168 * e1**4 * e2**2 * th * cos(2 * (g1 -
      g2)) - 588 * e1**2 * e2**2 * th**2 * cos(2 * (g1 - g2)) - 294 * e1**4
      * e2**2 * th**2 * cos(2 * (g1 - g2)) - 784 * e1**2 * e2**2 * th**3 *
      cos(2 * (g1 - g2)) - 392 * e1**4 * e2**2 * th**3 * cos(2 * (g1 - g2))
      - 128 * e2**2 * th * cos(2 * g2) - 640 * e1**2 * e2**2 * th * cos(2 *
      g2) - 240 * e1**4 * e2**2 * th * cos(2 * g2) + 224 * e2**2 * th**3 *
      cos(2 * g2) + 1120 * e1**2 * e2**2 * th**3 * cos(2 * g2) + 420 *
      e1**4 * e2**2 * th**3 * cos(2 * g2) - 140 * e1**2 * e2**2 * cos(2 *
      (g1 + g2)) - 70 * e1**4 * e2**2 * cos(2 * (g1 + g2)) + 336 * e1**2 *
      e2**2 * th * cos(2 * (g1 + g2)) + 168 * e1**4 * e2**2 * th * cos(2 *
      (g1 + g2)) + 588 * e1**2 * e2**2 * th**2 * cos(2 * (g1 + g2)) + 294 *
      e1**4 * e2**2 * th**2 * cos(2 * (g1 + g2)) - 784 * e1**2 * e2**2 *
      th**3 * cos(2 * (g1 + g2)) - 392 * e1**4 * e2**2 * th**3 * cos(2 *
      (g1 + g2)) + 147 * e1**4 * e2**2 * cos(2 * (2 * g1 + g2)) - 441 *
      e1**4 * e2**2 * th**2 * cos(2 * (2 * g1 + g2)) + 294 * e1**4 * e2**2
      * th**3 * cos(2 * (2 * g1 + g2))) + 1 / (e1 * sqrt((1 - e2**2)**7)) *
      2 * (1 - e1**2) * (m0 + m1) * (m0**3 + m1**3) * m2 * (e1 * (4 + 3 *
      e1**2) * (2 + 3 * e2**2) * (3 - 30 * th**2 + 35 * th**4) - 28 * (e1
      + e1**3) * (2 + 3 * e2**2) * (1 - 8 * th**2 + 7 * th**4) * cos(2 *
      g1) + 147 * e1**3 * (2 + 3 * e2**2) * (th**2 - 1)**2 * cos(4 * g1) -
      10 * e1 * (4 + 3 * e1**2) * e2**2 * (1 - 8 * th**2 + 7 * th**4) *
      cos(2 * g2) + 28 * (e1 + e1**3) * e2**2 * ((1 + th)**2 * (1 - 7 * th
      + 7 * th**2) * cos(2 * (g1 - g2)) + (th - 1)**2 * (1 + 7 * th + 7 *
      th**2) * cos(2 * (g1 + g2))) - 147 * e1**3 * e2**2 * (th**2 - 1) *
      ((1 + th)**2 * cos(4 * g1 - 2 * g2) + (th - 1)**2 * cos(2 * (2 * g1 +
      g2))))))

  de1dt = (30 * C2 * e1 * (1 - e1**2) / G1 * (1 - th**2) * sin(2 * g1) - C3 
    * e2 * (1 - e1**2) / G1 * (35 * cosphi * (1 - th**2) * e1**2 * sin(2 *
    g1) - 10 * th * (1 - e1**2) * (1 - th**2) * cosg1 * sing2 - A * (sing1
    * cosg2 - th * cosg1 * sing2))) 
  if input_gr:
    de1dt += (- 304 * G**3 * m0 * m1 * (m0 + m1) * e1
    / (15 * c*c*c*c*c * a1**4 * sqrt((1 - e1**2)**5)) * (1 + 121 / 304. *
    e1**2))
  if input_reo:
    if e1 != 0:
      de1dt += -(15 / 4. * sqrt(G * (m0 + m1) / a1**3) * (a1 / a2)**3 * th * 
          sqrt(1 - e1**2) / e1 * sin(G * (m0 + m1 + m2) / a2**3 * t +
          phase0))

  if input_hex:
    de1dt += (-(315 * a1**3 * e1 * sqrt(1 - e1**2) * sqrt(a1 * G * (m0 +
    m1)) * (m0**2 - m0 * m1 + m1**2) * m2 * (2 * (2 + e1**2) * (2 + 3 *
    e2**2) * (1 - 8 * th**2 + 7 * th**4) * sin(2 * g1) - 21 * e1**2 * (2 +
    3 * e2**2) * (th**2 - 1)**2 * sin(4 * g1) + e2**2 * (21 * e1**2 * (th -
    1) * (1 + th)**3 * sin(4 * g1 - 2 * g2) - 2 * (2 + e1**2) * (1 + th)**2
    * (1 - 7 * th + 7 * th**2) * sin(2 * (g1 - g2)) - (th - 1)**2 * (2 * (2
    + e1**2) * (1 + 7 * th + 7 * th**2) * sin(2 * (g1 + g2)) - 21 * e1**2 *
    (th**2 - 1) * sin(2 * (2 * g1 + g2)))))) / (2048 * a2**5 * sqrt((1 -
    e2**2)**7) * (m0 + m1)**3))

  da2dt = 0.0
  
  dg2dt = 0.0
  if e2 != 0.0:
    dg2dt += (3 * C2 * (2 * th / G1 * (2 + e1**2 * (3 - 5 * cos(2 * g1))) + 1
      / G2 * (4 + 6 * e1**2 + (5 * th**2 - 3) * (2 + 3 * e1**2 - 5 * e1**2 *
      cos(2 * g1)))) - C3 * e1 * sing1 * sing2 * ((4 * e2**2 + 1) / (e2 * G2)
      * 10 * th * (1 - th**2) * (1 - e1**2) - e2 * (1 / G1 + th / G2) * (A +
      10 * (3 * th**2 - 1) * (1 - e1**2))) - C3 * e1 * cosphi * (5 * B * th
      * e2 * (1 / G1 + th / G2) + (4 * e2**2 + 1) / (e2 * G2) * A))

    if input_hex:
      dg2dt += ((9 * a1**3 * (-1 / sqrt(1 - e1**2) * 10 * a2 * sqrt(a1 * G *
      (m0 + m1)) * (m0**2 - m0 * m1 + m1**2) * (sqrt(1 - e2**2) * m1**2 * m2
      * sqrt(a2 * G * (m0 + m1 + m2)) + m0**2 * (sqrt(1 - e2**2) * m2 *
      sqrt(a2 * G * (m0 + m1 + m2)) + sqrt(1 - e1**2) * m1 * sqrt(a1 * G *
      (m0 + m1)) * th) + m0 * m1 * (2 * sqrt(1 - e2**2) * m2 * sqrt(a2 * G *
      (m0 + m1 + m2)) + sqrt(1 - e1**2) * m1 * sqrt(a1 * G * (m0 + m1)) * th
      + sqrt(1 - e1**2) * sqrt(a1 * G * (m0 + m1)) * m2 * th)) * (96 * th +
      480 * e1**2 * th + 180 * e1**4 * th + 144 * e2**2 * th + 720 * e1**2 *
      e2**2 * th + 270 * e1**4 * e2**2 * th - 224 * th**3 - 1120 * e1**2 *
      th**3 - 420 * e1**4 * th**3 - 336 * e2**2 * th**3 - 1680 * e1**2 *
      e2**2 * th**3 - 630 * e1**4 * e2**2 * th**3 + 56 * e1**2 * (2 + e1**2)
      * (2 + 3 * e2**2) * th * (7 * th**2 - 4) * cos(2 * g1) - 294 * e1**4 *
      (2 + 3 * e2**2) * th * (th**2 - 1) * cos(4 * g1) - 147 * e1**4 *
      e2**2 * cos(4 * g1 - 2 * g2) + 441 * e1**4 * e2**2 * th**2 *
      cos(4 * g1 - 2 * g2) + 294 * e1**4 * e2**2 * th**3 * cos(4 * g1 - 2 *
      g2) + 140 * e1**2 * e2**2 * cos(2 * (g1 - g2)) + 70 * e1**4 * e2**2 *
      cos(2 * (g1 - g2)) + 336 * e1**2 * e2**2 * th * cos(2 * (g1 - g2)) +
      168 * e1**4 * e2**2 * th * cos(2 * (g1 - g2)) - 588 * e1**2 * e2**2 *
      th**2 * cos(2 * (g1 - g2)) - 294 * e1**4 * e2**2 * th**2 * cos(2 * (g1
      - g2)) - 784 * e1**2 * e2**2 * th**3 * cos(2 * (g1 - g2)) - 392 * e1**4
      * e2**2 * th**3 * cos(2 * (g1 - g2)) - 128 * e2**2 * th * cos(2 * g2) -
      640 * e1**2 * e2**2 * th * cos(2 * g2) - 240 * e1**4 * e2**2 * th *
      cos(2 * g2) + 224 * e2**2 * th**3 * cos(2 * g2) + 1120 * e1**2 * e2**2
      * th**3 * cos(2 * g2) + 420 * e1**4 * e2**2 * th**3 * cos(2 * g2) - 140
      * e1**2 * e2**2 * cos(2 * (g1 + g2)) - 70 * e1**4 * e2**2 * cos(2 * (g1
      + g2)) + 336 * e1**2 * e2**2 * th * cos(2 * (g1 + g2)) + 168 * e1**4 *
      e2**2 * th * cos(2 * (g1 + g2)) + 588 * e1**2 * e2**2 * th**2 * cos(2
      * (g1 + g2)) + 294 * e1**4 * e2**2 * th**2 * cos(2 * (g1 + g2)) - 784 *
      e1**2 * e2**2 * th**3 * cos(2 * (g1 + g2)) - 392 * e1**4 * e2**2 *
      th**3 * cos(2 * (g1 + g2)) + 147 * e1**4 * e2**2 * cos(2 * (2 * g1 +
      g2)) - 441 * e1**4 * e2**2 * th**2 * cos(2 * (2 * g1 + g2)) + 294 *
      e1**4 * e2**2 * th**3 * cos(2 * (2 * g1 + g2))) + a1 * a2 * G * m0 * m1
      * (m0**3 + m1**3) * (m0 + m1 + m2) * (-6 * (8 + 40 * e1**2 + 15 *
      e1**4) * (-1 + e2**2) * (3 - 30 * th**2 + 35 * th**4) + 7 * (8 + 40 *
      e1**2 + 15 * e1**4) * (2 + 3 * e2**2) * (3 - 30 * th**2 + 35 * th**4)
      + 840 * e1**2 * (2 + e1**2) * (-1 + e2**2) * (1 - 8 * th**2 + 7 *
      th**4) * cos(2 * g1) - 980 * e1**2 * (2 + e1**2) * (2 + 3 * e2**2) * (1
      - 8 * th**2 + 7 * th**4) * cos(2 * g1) - 4410 * e1**4 * (-1 + e2**2) *
      (-1 + th**2)**2 * cos(4 * g1) + 5145 * e1**4 * (2 + 3 * e2**2) * (-1 +
      th**2)**2 * cos(4 * g1) - 70 * (8 + 40 * e1**2 + 15 * e1**4) * e2**2 *
      (1 - 8 * th**2 + 7 * th**4) * cos(2 * g2) + 20 * (8 + 40 * e1**2 + 15 *
      e1**4) * (-1 + e2**2) * (1 - 8 * th**2 + 7 * th**4) * cos(2 * g2) + 980
      * e1**2 * (2 + e1**2) * e2**2 * ((1 + th)**2 * (1 - 7 * th + 7 * th**2)
      * cos(2 * (g1 - g2)) + (-1 + th)**2 * (1 + 7 * th + 7 * th**2) * cos(2
      * (g1 + g2))) - 280 * e1**2 * (2 + e1**2) * (-1 + e2**2) * ((1 + th)**2
      * (1 - 7 * th + 7 * th**2)  * cos(2 * (g1 - g2)) + (-1 + th)**2 * (1 +
      7 * th + 7 * th**2) * cos(2 * (g1 + g2))) - 1470 * e1**4 * (1 - e2**2)
      * (-1 + th) * (1 + th) * ((1 + th)**2 * cos(4 * g1 - 2 * g2) + (-1 +
      th)**2 * cos(2 * (2 * g1 + g2))) - 5145 * e1**4 * e2**2 * (-1 + th**2)
      * ((1 + th)**2 * cos(4 * g1 - 2 * g2) + (-1 + th)**2 * cos(2 * (2 * g1
      + g2)))))) / (8192 * a2**6 * (-1 + e2**2)**4 * (m0 + m1)**5 * sqrt(a2 *
      G * (m0 + m1 + m2))))

  de2dt = (C3 * e1 * (1 - e2**2) / G2 * (10 * th * (1 - th**2) * (1 -
  e1**2) * sing1 * cosg2 + A * (cosg1 * sing2 - th * sing1 * cosg2)))

  if input_hex:
    de2dt += ((45 * a1**4 * e2 * m0 * m1 * (m0**2 - m0 * m1 + m1**2) *
      sqrt(a2 * G * (m0 + m1 + m2)) * (-147 * e1**4 * (-1 + th) * (1 +
      th)**3 * sin(4 * g1 - 2 * g2) + 28 * e1**2 * (2 + e1**2) * (1 +
      th)**2 * (1 - 7 * th + 7 * th**2) * sin(2 * (g1 - g2)) + (-1 + th) *
      (2 * (8 + 40 * e1**2 + 15 * e1**4) * (-1 - th + 7 * th**2 + 7 *
      th**3) * sin(2 * g2) - 7 * e1**2 * (-1 + th) * (4 * (2 + e1**2) * (1
      + 7 * th + 7 * th**2) * sin(2 * (g1 + g2)) - 21 * e1**2 * (-1 + th**2)
      * sin(2 * (2 * g1 + g2))))) / (4096 * a2**6 * (-1 + e2**2)**3 * (m0 +
      m1)**4)))

  dHdt = 0.0
  if input_gr:
    dHdt += (-32 * G**3 * m0**2 * m1**2 / (5 * c*c*c*c*c * a1**3 * 
      (1 - e1**2)**2) * sqrt(G * (m0 + m1) / a1) * (1 + 7 / 8. * e1**2) * 
      (G1 + G2 * th) / H)

  der = (da1dt, dg1dt, de1dt, da2dt, dg2dt, de2dt, dHdt)
  return der

def printout(t, y, m):
  '''Print out the state of the system.'''
  print t/auconsts.year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], 
  print calc_cosi(consts, m, y)   

def secular(m, r, e, a, g, inc, tstop, outfreq=-1, acc=(1e-13, 1e-13)):
  '''
  Evolve a hierarchical triple system using the secular approximation.
  
  Input:
    m -- A tuple containing the three masses in kg (m0, m1, m2)
    r -- A tuple containing the inner two radii in m (r0, r1)
    e -- A tuple containing the eccentricities (e1, e2)
    a -- A tuple containing the semi-major axes in m (a1, a2)
    g -- A tuple containing the arguments of periapsis (g1, g2)
    inc     -- The mutual inclination in radians
    tstop   -- A tuple containing the stop time and the CPU stop time in sec
               (t_stop, t_cpu_stop)
    outfreq -- How many steps between printing out the state
              (-1 for no printing)
    acc     -- A tuple containing the accuracy targets (relacc, absacc)

  Output:
    If outfreq is n, where in is not -1, the function will print the state
    of the system every n steps to stdout.

    The function will return a tuple containing:
      -- The final time
      -- The maximum eccentricity over the run
      -- A list containing the final state of the system, consisting of:
          o A tuple containing the eccentricities (e1, e2)
          o A tuple containing the semi-major axes in m (a1, a2)
          o g -- A tuple containing the arguments of periapsis (g1, g2)
          o inc -- The mutual inclination in radians
      -- A merger flag which is True if the system merged
      -- An error flag which is True if an exception was encountered
  '''

  # Unpack the inputs
  m0, m1, m2 = m
  e1, e2 = e
  a1, a2 = a
  g1, g2 = g
  endtime, cputime = tstop
  relacc, absacc = acc

  consts = (auconsts.G, auconsts.c) # G, c

  # Set up the GSL integrator
  dimension = 7

  # The different possible steppers for the function
  # Comment your favorite one out to use it.
  #stepper = odeiv.step_rk2
  #stepper = odeiv.step_rk4
  #stepper = odeiv.step_rkf45
  #stepper = odeiv.step_rkck
  stepper = odeiv.step_rk8pd
  #stepper = odeiv.step_rk2imp
  #stepper = odeiv.step_rk4imp
  #stepper = odeiv.step_gear1
  #stepper = odeiv.step_gear2

  step = stepper(dimension, deriv)
  control = odeiv.control_y_new(step, absacc, relacc)
  evolve  = odeiv.evolve(step, control, dimension)

  # 0  1  2  3  4  5  6
  # a1 g1 e1 a2 g2 e2 H
  yinit = (a1, g1, e1, a2, g2, e2, calc_H(e, inc, consts, m, a))

  #initial step size h, it will be adapted by apply
  time_step = 1000.0 * auconsts.year
  tinit = 0.0

  t = tinit
  y = yinit
  yprev = y[:]
  stamp = time.time()
  count = 0
  maxe = e1
  merge_flag = False
  exception_flag = False

  # Print the initial conditions (if we're printing at all)
  if outfreq != -1:
    printout(t, y, m)

  while (t < endtime):
    # Stop if the CPU time limit is reached
    if ((time.time() - stamp) >  cputime):
      if outfreq != -1:
        printout(t, y, m)
      
      raise ValueError, "cpu limit reached!"

    yprev = y[:]

    try:
      t, time_step, y = evolve.apply(t, endtime, time_step, y)
    except ValueError:
      exception_flag = True
      y = yprev[:] # the current y might be nonsensical
      break

    y[1] = y[1] % (2 * pi)
    y[4] = y[4] % (2 * pi)

    if y[2] > maxe:
      maxe = y[2]
    
    if outfreq != -1:
      if (count % outfreq == 0):
        printout(t, y, m)
    count += 1
          
    if (y[0] < r0 + r1):
      merge_flag = True
      break

  printout(t, y, m)
  e = (e1, e2)
  a = (a1, a2)
  g = (g1, g2)
  return (t, maxe, (e, a, g, calc_cosi(consts, m, y)), merger_flag,
    exception_flag)

if __name__=="__main__":
  import optparse
  import sys
  import astroutils.consts as auconsts

  # Defualt parameters
  def_a1 = 10
  def_a2 = 250
  def_g1 = 0.
  def_g2 = 0.
  def_e1 = .3
  def_e2 = .3
  def_inc = 80
  def_endtime = 1e7
  def_cputime = 300.0
  def_m0 = 7.0
  def_m1 = 7.0
  def_m2 = 6.0
  def_outfreq = 1e2
  def_absacc = 1e-13
  def_relacc = 1e-13
  def_verb = 0
  def_oct = 1
  def_gr = 1
  def_reo = 0
  def_phase = -1
  def_hex = 1

  # Configure the command line options
  parser = optparse.OptionParser()
  parser.add_option('-m', '--m00', dest="m0", type = "float", default=def_m0,
    help = "Mass of first object in inner binary in solar masses [%g]" %
    def_m0)
  parser.add_option('-n', '--m01', dest="m1", type = "float", default=def_m1,
    help = "Mass of second object in inner binary in solar masses [%g]" %
    def_m1)
  parser.add_option('-o', '--m1', dest="m2", type = "float", default=def_m2,
    help = "Mass of tertiary in solar masses [%g]" % def_m2)
  parser.add_option('-r', '--r00', dest="r0", type = "float", default=def_r0,
    help = "Radius of star 1 of the inner binary in R_Sun [%g]" % def_r0)
  parser.add_option('-s', '--r01', dest="r1", type = "float", default=def_r1,
    help = "Radius of star 2 of the inner binary in R_Sun [%g]" % def_r1)
  parser.add_option('-a', '--a00', dest="a1", type="float", default=def_a1,
    help = "Inner semi-major axis in au [%g]" % def_a1)
  parser.add_option('-b', '--a0', dest="a2", type="float", default=def_a2, 
    help = "Outer semi-major axis in au [%g]" % def_a2)
  parser.add_option('-g', '--g00', dest="g1", type="float", default=def_g1,
    help = "Inner argument of periapsis in degrees [%g]" % def_g1)
  parser.add_option('-G', '--g0', dest="g2", type="float", default=def_g2,
    help = "Outer argument of periapsis in degrees [%g]" % def_g2)
  parser.add_option('-e', '--e00', dest="e1", type="float", default=def_e1,
    help = "Inner eccentricity [%g]" % def_e1)
  parser.add_option('-f', '--e0', dest="e2", type="float", default=def_e2,
    help = "Outer eccentricity [%g]" % def_e2)
  parser.add_option('-i', '--inc', dest="inc", type="float",
    default=def_inc, help = "Inclination of the third body in degrees [%g]" %
    def_inc)
  parser.add_option('-T', '--end', dest="endtime", type="float", default=def_endtime,
    help = "Total time of integration in years [%g]" % def_endtime)
  parser.add_option('-C', '--cpu', dest="cputime", type="float", default=def_cputime,
    help = "cpu time limit in seconds, if -1 then no limit [%g]" % def_cputime)
  parser.add_option('-F', '--freq', dest="outfreq", type = "int", default=def_outfreq,
    help = "Output frequency [%g]" % def_outfreq)
  parser.add_option('-A', '--aacc', dest="absacc", type = "float", default=def_absacc,
    help = "Absolute Accuracy [%g]" % def_absacc)
  parser.add_option('-R', '--racc', dest="relacc", type = "float", default=def_relacc,
    help = "Relative Accuracy [%g]" % def_relacc)
  parser.add_option('-v', '--verb', dest="verb", type = "int", default=def_verb,
    help = "Verbose [%g]" % def_verb)
  parser.add_option('-O', '--oct', dest="oct", type = "int", default=def_oct,
    help = "Octupole terms [%g]" % def_oct)
  parser.add_option('-c', '--GR', dest="gr", type = "int", default=def_gr,
    help = "General Relativity [%g]" % def_gr)
  parser.add_option('-z', '--reo', dest="reo", type = "int", default=def_reo,
    help = "Reo [%g]" % def_reo)
  parser.add_option('-p', '--phase', dest="phase", type = "float", default=def_phase,
    help = "Initial phase [%g]" % def_phase)
  parser.add_option('-x', '--hex', dest="hex", type = "int", default=def_hex,
    help = "Hexadecapole [%g]" % def_hex)

  # Read in the command line
  options, remainder = parser.parse_args()

  # Check for problems on the command line
  if options.m0 < 0:
    print >> sys.stderr, "m0 must be greater than zero."
    sys.exit(1)
  m0 = options.m0 * auconsts.msun
  if options.m1 < 0:
    print >> sys.stderr, "m1 must be greater than zero."
    sys.exit(1)
  m1 = options.m1 * auconsts.msun
  if options.m2 < 0:
    print >> sys.stderr, "m2 must be greater than zero."
    sys.exit(1)
  m2 = options.m2 * auconsts.msun
  if options.r0 < 0:
    print >> sys.stderr, "r0 must be greater than zero."
    sys.exit(1)
  r0 = options.r0 * auconsts.rsun
  if options.r1 < 0:
    print >> sys.stderr, "r1 must be greater than zero."
    sys.exit(1)
  r1 = options.r1 * auconsts.rsun
  if options.a1 < 0:
    print >> sys.stderr, "a1 must be greater than zero."
    sys.exit(1)
  a1 = options.a1 * auconsts.au
  if options.a2 < 0:
    print >> sys.stderr, "a2 must be greater than zero."
    sys.exit(1)
  a2 = options.a2 * auconsts.au
  if options.g1 < 0 or options.g1 >= 360:
    print >> sys.stderr, "g1 must be between 0 and 360."
    sys.exit(1)
  g1 = options.g1 * pi / 180.
  if options.g2 < 0 or options.g2 >= 360:
    print >> sys.stderr, "g2 must be between 0 and 360."
    sys.exit(1)
  g2 = options.g2 * pi / 180.
  if options.e1 < 0 or options.e1 > 1:
    print >> sys.stderr, "e1 must be between 0 and 1."
    sys.exit(1)
  e1 = options.e1
  if options.e2 < 0 or options.e2 > 1:
    print >> sys.stderr, "e2 must be between 0 and 1."
    sys.exit(1)
  e2 = options.e2
  if options.relacc > 1 or options.relacc < 0:
    print >> sys.stderr, "relacc must be between 0 and 1."
    sys.exit(1)
  relacc = options.relacc
  if options.absacc < 0 or options.absacc > 1:
    print >> sys.stderr, "absacc must be between 0 and 1."
    sys.exit(1)
  absacc = options.absacc
  if options.endtime < 0:
    print >> sys.stderr, "stop time must be greater than 0"
    sys.exit(1)
  endtime = sys.endtime
  if options.cputime < 0 and options.cputime != -1:
    print >> sys.stderr, "cputime must greater than 0 of exactly -1."
    sys.exit(1)
  cputime = options.cputime
  if options.inc < -90 or options.inc > 90:
    print >> sys.stderr, "cosi must be between -90 and 90 degrees."
    sys.exit(1)
  inc = options.inc * pi / 180.
  if options.outfreq < 0 and options.outfreq != -1:
    print >> sys.stderr, "output frequency must be greater than zero or exactly equal to -1"
    sys.exit(1)
  outfreq = options.outfreq
  if options.verb != 0 and options.verb != 1:
    print >> sys.stderr, "verbose must either be on (1) or off (0)"
    sys.exit(1)
  verb = options.verb
  if options.oct != 0 and options.oct != 1:
    print >> sys.stderr, "octupole terms must either be on (1) or off (0)"
    sys.exit(1)
  oct = options.oct
  if options.gr != 0 and options.gr != 1:
    print >> sys.stderr, "general relativity must either be on (1) or off (0)"
    sys.exit(1)
  gr = options.gr
  if options.reo != 0 and options.reo != 1:
    print >> sys.stderr, "reos must be either on (1) or off (0)"
    sys.exit(1)
  reo = options.reo
  if options.phase > 2 * pi or (options.phase < 0 and options.phase != -1):
    print >> sys.stderr, "Initial phase must be between 0 and 2pi or exactly -1 (for random)"
    sys.exit(1)
  phase = options.phase
  if options.hex != 0 and options.hex != 1:
    print >> sys.stderr, "hexadecapole term must be either on (1) or off (0)"
    sys.exit(1)
  hex = options.hex

  if phase == -1:
    phase0 = random.uniform(0, 2 * pi)
  else:
    phase0 = phase

  if cputime == -1:
    cputime = float('inf')

  m = (m0, m1, m2)

  print >> sys.stderr, "Secular evolution calculation"
  print >> sys.stderr
  print >> sys.stderr, "System parameters"
  print >> sys.stderr, "a1 = ", a1, "a2 = ", a2
  print >> sys.stderr, "m0 = ", m0, "m1 = ", m1, "m2 = ", m2
  print >> sys.stderr, "e1 = ", e1, "e2 = ", e2
  print >> sys.stderr, "g1 = ", g1, "g2 = ", g2
  print >> sys.stderr, "inc = ", inc
  print >> sys.stderr, "t_final = ", endtime, "outfreq = ", outfreq
  print >> sys.stderr, "reo = ", input_reo, "hex = ", input_hex, "phase = ", input_phase
  print >> sys.stderr

  print >> sys.stderr, yinit
  print >> sys.stderr, endtime
  print >> sys.stderr

  # Parameters to give to the secular function
  m = (m0, m1, m2)
  r = (r0, r1)
  e = (e1, e2)
  a = (a1, a2)
  g = (g1, g2)
  tstop = (endtime, cputime)
  acc = (relacc, absacc)

  # Run the secular calculation
  t, maxe, state, merger_flag, exception_flag = secular(m, r, e, a, g, inc, tstop, acc)

  # TODO: Recover information about the halt criterion -- did we stop
  # because of tcpu, endtime, or exception?

  e, a, g, inc = state
  e1, e2 = e
  a1, a2 = a
  g1, g2 = g

  # Print the results of the calculation
  print >> sys.stderr
  print >> sys.stderr, "Calculation complete"
  print >> sys.stderr, "t =", t
  print >> sys.stderr, "maxe =", maxe

  print >> sys.stderr, "Merger =",
  if merger_flag:
    print >> sys.stderr, "True"
  else:
    print >> sys.stderr, "False"

  print >> sys.stderr, "Exception =",
  if exception_flag:
    print >> sys.stderr, "True"
  else:
    print >> sys.stderr, "False"

  if verb == 1:
    print >> sys.stderr, "# Using stepper %s with order %d" %(step.name(), step.order())
    print >> sys.stderr, "# Using Control ", control.name()
    print >> sys.stderr, yinit
    print >> sys.stderr, endtime, time_step 
    print >> sys.stderr, "Needed %f seconds" %( time.time() - stamp,)
      
