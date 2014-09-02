#! /usr/bin/env python 

#
# secular
#
# Numerically integrates Equations 11 -- 17 in Blaes, Lee, & Socrates
# (2002).  Now includes the hexadecapole term.
#

from math import sqrt, cos, sin, pi, acos
import numpy as np
import time
from pygsl import odeiv
import random
import astropy.constants as const
import astropy.units as unit

def calc_H(e, i, m, a):
  '''Calculates H.  See eq. 22 of Blaes et al. (2002)'''

  a1, a2 = a
  e1, e2 = e
  G = const.G.value
  c = const.c.value
  m0, m1, m2 = m

  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))
  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  return sqrt(2 * cos(i) * G1 * G2 + G1**2 + G2**2)

def calc_cosi(m, y):
  '''Calculates the cosine of the inclination given H.'''

  G = const.G.value

  a1 = y[0]
  e1 = y[2]
  a2 = y[3]
  e2 = y[5]
  H = y[6]

  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))
  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  return (H**2 - G1**2 - G2**2) / (2 * G1 * G2)

def deriv(t, y, in_params):
  '''Compute derivatives of y at t'''

  # Unpack the values
  a1, g1, e1, a2, g2, e2, H = y
  G = const.G.value
  c = const.c.value
  m0, m1, m2 = m
  input_gr, input_oct, input_hex = in_params

  # Calculate trig functions only once
  sing1 = sin(g1)
  sing2 = sin(g2)
  cosg1 = cos(g1)
  cosg2 = cos(g2)

  C2 = (G * m0 * m1 * m2 / (16 * (m0 + m1) * a2 * sqrt((1 - e2**2)**3)) *
    (a1 / a2)**2)

  if input_oct:
      C3 = (15 * G * m0 * m1 * m2 * (m0 - m1) / (64 * (m0 + m1)**2 * a2 * 
        sqrt((1 - e2**2)**5)) * (a1 / a2)**3)
  else:
    C3 = 0.

  if not input_gr:
    c = np.inf
  
  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))
  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  th = (H**2 - G1**2 - G2**2) / (2 * G1 * G2)
  cosphi = - cosg1 * cosg2 - th * sing1 * sing2

  B = 2 + 5 * e1**2 - 7 * e1**2 * cos(2 * g1)
  A = 4 + 3 * e1**2 - 5 / 2. * (1 - th**2) * B

  # Eq. 11 of Blaes et al. (2002)
  da1dt = -(64 * G**3 * m0 * m1 * (m0 + m1) / (5 * c**5 * a1**3 * 
    sqrt((1 - e1**2)**7)) * (1 + 73 / 24. * e1**2 + 37 / 96. * e1**4))

  # Eq. 12 of Blaes et al. (2002)
  dg1dt = (6 * C2 * (1 / G1 * (4 * th**2 + (5 * cos(2 * g1) - 1) * (1 -
    e1**2 - th**2)) + th / G2 * (2 + e1**2 * (3 - 5 * cos(2 * g1)))) + C3 *
    e2 * e1 * (1 / G2 + th / G1) * (sing1 * sing2 * (A + 10 * (3 *
    th**2 - 1) * (1 - e1**2)) - 5 * th * B * cosphi) - C3 * e2 * (1 -
    e1**2) / (e1 * G1) * (10 * th * (1 - th**2) * (1 - 3 * e1**2) * sing1
    * sing2 + cosphi * (3 * A - 10 * th**2 + 2)) + (3 / (c**2 * a1 * (1 -
    e1**2)) * sqrt((G * (m0 + m1) / a1)**3)))

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

  # Eq. 13 of Blaes et al. (2002)
  de1dt = (30 * C2 * e1 * (1 - e1**2) / G1 * (1 - th**2) * sin(2 * g1) - C3 
    * e2 * (1 - e1**2) / G1 * (35 * cosphi * (1 - th**2) * e1**2 * sin(2 *
    g1) - 10 * th * (1 - e1**2) * (1 - th**2) * cosg1 * sing2 - A * (sing1
    * cosg2 - th * cosg1 * sing2)) - 304 * G**3 * m0 * m1 * (m0 + m1) * e1
    / (15 * c**4 * a1**4 * sqrt((1 - e1**2)**5)) * (1 + 121 / 304. *
    e1**2))

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

  da2dt = 0.
  
  dg2dt = (3 * C2 * (2 * th / G1 * (2 + e1**2 * (3 - 5 * cos(2 * g1))) + 1
      / G2 * (4 + 6 * e1**2 + (5 * th**2 - 3) * (2 + 3 * e1**2 - 5 * e1**2 *
      cos(2 * g1))))) 
  if e2 != 0:
    dg2dt += (-C3 * e1 * sing1 * sing2 * ((4 * e2**2 + 1) / (e2 * G2) * 10 * 
      th * (1 - th**2) * (1 - e1**2) - e2 * (1 / G1 + th / G2) * (A + 10 * 
      (3 * th**2 - 1) * (1 - e1**2))) - C3 * e1 * cosphi * (5 * B * th * 
      e2 * (1 / G1 + th / G2) + (4 * e2**2 + 1) / (e2 * G2) * A))

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

  # Eq. 16 of Blaes et al. (2002)
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

  # Eq. 17 of Blaes et al. (2002)
  dHdt = (-32 * G**3 * m0**2 * m1**2 / (5 * c**5 * a1**3 * (1 - e1**2)**2) 
    * sqrt(G * (m0 + m1) / a1) * (1 + 7 / 8. * e1**2) * (G1 + G2 * th) 
    / H)

  der = (da1dt, dg1dt, de1dt, da2dt, dg2dt, de2dt, dHdt)
  return der

def printout(t, y, m):
  '''Print out the state of the system.'''
  print t / (unit.year / unit.s).to(1), 
  print y[0] / const.au.value, y[1], y[2], 
  print y[3] / const.au.value, y[4], y[5], y[6], calc_cosi(m, y)

def secular(m, r, e, a, g, inc, tstop, 
  in_params=(1, (1e-13, 1e-13), (False, True, False))):
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
    in_params -- A tuple containing:
      outfreq: How many steps between printing out the state
               (-1 for no printing)
      acc: A tuple containing the accuracy targets (relacc, absacc)
      terms: A tuple saying what terms to include (gc, oct, hex)

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
  outfreq, acc, terms = in_params
  relacc, absacc = acc

  in_params = (input_gr, input_oct, input_hex)

  # 0  1  2  3  4  5  6
  # a1 g1 e1 a2 g2 e2 H
  yinit = (a1, g1, e1, a2, g2, e2, calc_H(e, inc, m, a))

  # Initial step size is taken to be an outer period.  It will be adapted by
  # GSL.
  time_step = 2 * pi * sqrt(a2**3 / (const.G.value * (m0 + m1 + m2)))

  # Set up the GSL integrator
  dimension = len(yinit)
  stepper = odeiv.step_rk8pd
  step = stepper(dimension, deriv, args=in_params)
  control = odeiv.control_y_new(step, absacc, relacc)
  evolve  = odeiv.evolve(step, control, dimension)

  t = 0.
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
      raise RuntimeError, 'secular(): CPU time limit reached!'

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
          
    if (y[0] * (1 - y[2]) <= r0 + r1):
      merge_flag = True
      break

  printout(t, y, m)
  e = (e1, e2)
  a = (a1, a2)
  g = (g1, g2)
  return (t, maxe, (e, a, g, calc_cosi(m, y)), merge_flag,
    exception_flag)

if __name__=='__main__':
  import optparse
  import sys
  import astropy.constants as const

  # Defualt parameters
  def_a1 = 1.
  def_a2 = 10.
  def_g1 = 0.
  def_g2 = 0.
  def_e1 = 0.
  def_e2 = 0.
  def_inc = 80
  def_endtime = 1e7
  def_cputime = 300.
  def_m0 = 1.
  def_m1 = 1.
  def_m2 = 1.
  def_r0 = 1.
  def_r1 = 1.
  def_outfreq = 1
  def_absacc = 1e-13
  def_relacc = 1e-13

  def_verb = False
  def_oct = True
  def_hex = False
  def_gr = True

  # Configure the command line options
  parser = optparse.OptionParser()
  parser.add_option('-m', '--m00', dest='m0', type = 'float', default=def_m0,
    help = 'Mass of star 1 in inner binary in solar masses [%g]' % def_m0)
  parser.add_option('-n', '--m01', dest='m1', type = 'float', default=def_m1,
    help = 'Mass of star 2 in inner binary in solar masses [%g]' % def_m1)
  parser.add_option('-o', '--m1', dest='m2', type = 'float', default=def_m2,
    help = 'Mass of tertiary in solar masses [%g]' % def_m2)
  parser.add_option('-r', '--r00', dest='r0', type = 'float', default=def_r0,
    help = 'Radius of star 1 of the inner binary in R_Sun [%g]' % def_r0)
  parser.add_option('-s', '--r01', dest='r1', type = 'float', default=def_r1,
    help = 'Radius of star 2 of the inner binary in R_Sun [%g]' % def_r1)
  parser.add_option('-a', '--a00', dest='a1', type='float', default=def_a1,
    help = 'Inner semi-major axis in au [%g]' % def_a1)
  parser.add_option('-b', '--a0', dest='a2', type='float', default=def_a2, 
    help = 'Outer semi-major axis in au [%g]' % def_a2)
  parser.add_option('-g', '--g00', dest='g1', type='float', default=def_g1,
    help = 'Inner argument of periapsis in degrees [%g]' % def_g1)
  parser.add_option('-G', '--g0', dest='g2', type='float', default=def_g2,
    help = 'Outer argument of periapsis in degrees [%g]' % def_g2)
  parser.add_option('-e', '--e00', dest='e1', type='float', default=def_e1,
    help = 'Inner eccentricity [%g]' % def_e1)
  parser.add_option('-f', '--e0', dest='e2', type='float', default=def_e2,
    help = 'Outer eccentricity [%g]' % def_e2)
  parser.add_option('-i', '--inc', dest='inc', type='float',
    default=def_inc, help = 'Inclination of the third body in degrees [%g]' %
    def_inc)
  parser.add_option('-t', '--end', dest='endtime', type='float', 
    default=def_endtime, help = 'Total time of integration in years [%g]' 
    % def_endtime)
  parser.add_option('-C', '--cpu', dest='cputime', type='float', 
    default=def_cputime, help = 
    'cpu time limit in seconds, if -1 then no limit [%g]' % def_cputime)
  parser.add_option('-F', '--freq', dest='outfreq', type = 'int', 
    default=def_outfreq, help = 'Output frequency [%g]' % def_outfreq)
  parser.add_option('-A', '--aacc', dest='absacc', type = 'float', 
    default=def_absacc, help = 'Absolute Accuracy [%g]' % def_absacc)
  parser.add_option('-R', '--racc', dest='relacc', type = 'float', 
    default=def_relacc, help = 'Relative Accuracy [%g]' % def_relacc)
  parser.add_option('-v', '--verb', action='store_true', dest='verb', 
    default = def_verb, help = 'Verbose')
  parser.add_option('--nooct', dest='oct', action='store_false',
    default = def_oct, help = 'Turn off octupole terms')
  parser.add_option('-c', '--GR', dest='gr', action='store_true', 
    default = def_gr, help = 'Turn on general relativity terms')
  parser.add_option('-x', '--hex', dest='hex', action='store_true',
    default = def_hex, help = 'Turn on hexadecapole terms')

  # Read in the command line
  options, remainder = parser.parse_args()

  # Check for problems on the command line and convert all units to MKS
  if options.m0 < 0:
    print >> sys.stderr, 'm0 must be greater than zero.'
    sys.exit(1)
  m0 = options.m0 * const.M_sun.value
  if options.m1 < 0:
    print >> sys.stderr, 'm1 must be greater than zero.'
    sys.exit(1)
  m1 = options.m1 * const.M_sun.value
  if options.m2 < 0:
    print >> sys.stderr, 'm2 must be greater than zero.'
    sys.exit(1)
  m2 = options.m2 * const.M_sun.value
  if options.r0 < 0:
    print >> sys.stderr, 'r0 must be greater than zero.'
    sys.exit(1)
  r0 = options.r0 * const.R_sun.value
  if options.r1 < 0:
    print >> sys.stderr, 'r1 must be greater than zero.'
    sys.exit(1)
  r1 = options.r1 * const.R_sun.value
  if options.a1 < 0:
    print >> sys.stderr, 'a1 must be greater than zero.'
    sys.exit(1)
  a1 = options.a1 * const.au.value
  if options.a2 < 0:
    print >> sys.stderr, 'a2 must be greater than zero.'
    sys.exit(1)
  a2 = options.a2 * const.au.value
  if options.g1 < 0 or options.g1 >= 360:
    print >> sys.stderr, 'g1 must be between 0 and 360.'
    sys.exit(1)
  g1 = options.g1 * pi / 180.
  if options.g2 < 0 or options.g2 >= 360:
    print >> sys.stderr, 'g2 must be between 0 and 360.'
    sys.exit(1)
  g2 = options.g2 * pi / 180.
  if options.e1 < 0 or options.e1 > 1:
    print >> sys.stderr, 'e1 must be between 0 and 1.'
    sys.exit(1)
  e1 = options.e1
  if options.e2 < 0 or options.e2 > 1:
    print >> sys.stderr, 'e2 must be between 0 and 1.'
    sys.exit(1)
  e2 = options.e2
  if options.relacc > 1 or options.relacc < 0:
    print >> sys.stderr, 'relacc must be between 0 and 1.'
    sys.exit(1)
  relacc = options.relacc
  if options.absacc < 0 or options.absacc > 1:
    print >> sys.stderr, 'absacc must be between 0 and 1.'
    sys.exit(1)
  absacc = options.absacc
  if options.endtime < 0:
    print >> sys.stderr, 'stop time must be greater than 0'
    sys.exit(1)
  endtime = options.endtime * (unit.yr / unit.s).to(1)
  if options.cputime < 0 and options.cputime != -1:
    print >> sys.stderr, 'cputime must greater than 0 of exactly -1.'
    sys.exit(1)
  cputime = options.cputime
  if options.inc < -90 or options.inc > 90:
    print >> sys.stderr, 'cosi must be between -90 and 90 degrees.'
    sys.exit(1)
  inc = options.inc * pi / 180.
  if options.outfreq < 0 and options.outfreq != -1:
    print >> sys.stderr, 'output frequency must be greater than zero or exactly equal to -1'
    sys.exit(1)
  outfreq = options.outfreq
  if options.verb != 0 and options.verb != 1:
    print >> sys.stderr, 'verbose must either be on (1) or off (0)'
    sys.exit(1)
  verb = options.verb
  input_oct = options.oct
  input_gr = options.gr
  input_hex = options.hex

  if cputime == -1:
    cputime = float('inf')

  print >> sys.stderr, 'Secular evolution calculation'
  print >> sys.stderr
  print >> sys.stderr, 'System parameters'
  print >> sys.stderr, 'a1 =', a1 / const.au.value, 
  print >> sys.stderr, 'a2 =', a2 / const.au.value
  print >> sys.stderr, 'm0 =', m0 / const.M_sun.value, "M_Sun",
  print >> sys.stderr, 'm1 =', m1 / const.M_sun.value, "M_Sun",
  print >> sys.stderr, 'm2 =', m2 / const.M_sun.value, "M_Sun"
  print >> sys.stderr, 'e1 =', e1, 'e2 =', e2
  print >> sys.stderr, 'g1 =', g1 * 180 / pi, 'g2 =', g2 * 180 / pi
  print >> sys.stderr, 'inc =', inc * 180 / pi
  print >> sys.stderr, 't_final =', endtime * (unit.s / unit.yr).to(1)
  print >> sys.stderr, 'outfreq =', outfreq
  print >> sys.stderr, 'gc =', input_gr, 'oct =', input_oct, 'hex =', input_hex
  print >> sys.stderr

  # Parameters to give to the secular function
  m = (m0, m1, m2)
  r = (r0, r1)
  e = (e1, e2)
  a = (a1, a2)
  g = (g1, g2)
  tstop = (endtime, cputime)
  acc = (relacc, absacc)

  in_params = (outfreq, acc, (input_gr, input_oct, input_hex))

  # Run the secular calculation
  try:
    t, maxe, state, merger_flag, exception_flag = secular(m, r, e, a, g, inc,
      tstop, in_params)
  except RuntimeError:
    pass

  # TODO: Recover information about the halt criterion -- did we stop
  # because of tcpu, endtime, or exception?

  e, a, g, inc = state
  e1, e2 = e
  a1, a2 = a
  g1, g2 = g

  # Print the results of the calculation
  print >> sys.stderr
  print >> sys.stderr, 'Calculation complete'
  print >> sys.stderr, 't =', t
  print >> sys.stderr, 'maxe =', maxe

  print >> sys.stderr, 'Merger =',
  if merger_flag:
    print >> sys.stderr, 'True'
  else:
    print >> sys.stderr, 'False'

  print >> sys.stderr, 'Exception =',
  if exception_flag:
    print >> sys.stderr, 'True'
  else:
    print >> sys.stderr, 'False'

  if verb == 1:
    print >> sys.stderr, '# Using stepper %s with order %d' %(step.name(), step.order())
    print >> sys.stderr, '# Using Control ', control.name()
    print >> sys.stderr, endtime, time_step 
    print >> sys.stderr, 'Needed %f seconds' %( time.time() - stamp,)
      
