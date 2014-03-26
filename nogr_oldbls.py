#! /usr/bin/env python 

#
# bls
#
# Numerically integrates Equations 11 -- 17 in Blaes, Lee, & Socrates
# (2002).  
#
# Joe Antognini
# Mon Jun 25 18:34:18 EDT 2012
#

from math import sqrt, cos, sin, pi
from numpy import linspace
import sys
import optparse
import random

def_a1 = 3.16e-3
def_a2 = 3.16e-2
def_g1 = 0.
def_g2 = 0.
def_e1 = .1
def_e2 = .1
def_inc = 80.
def_endtime = 1e16
def_tstep = 1e9
def_m0 = 2e6
def_m1 = 1e6
def_m2 = 1e6
def_outfreq = 1
def_reo = 1
def_phase = -1
def_hex = 1

parser = optparse.OptionParser()
parser.add_option('-a', dest="a1", type="float", default=def_a1,
  help = "Inner semi-major axis in parsecs [%g]" % def_a1)
parser.add_option('-b', dest="a2", type="float", default=def_a2, 
  help = "Outer semi-major axis in parsecs [%g]" % def_a2)
parser.add_option('-g', dest="g1", type="float", default=def_g1,
  help = "Inner argument of periapsis in degrees [%g]" % def_g1)
parser.add_option('-G', dest="g2", type="float", default=def_g2,
  help = "Outer argument of periapsis in degrees [%g]" % def_g2)
parser.add_option('-e', dest="e1", type="float", default=def_e1,
  help = "Inner eccentricity [%g]" % def_e1)
parser.add_option('-f', dest="e2", type="float", default=def_e2,
  help = "Outer eccentricity [%g]" % def_e2)
parser.add_option('-i', '--inc', '--inclination', dest="inc", type="float",
  default=def_inc, help = "Inclination of the third body in degrees [%g]" %
  def_inc)
parser.add_option('-T', dest="endtime", type="float", default=def_endtime,
  help = "Total time of integration in secends [%g]" % def_endtime)
parser.add_option('-t', dest="timestep", type="float", default=def_tstep,
  help = "Size of time step in secends [%g]" % def_tstep)
parser.add_option('-m', dest="m0", type = "float", default=def_m0,
  help = "Mass of first object in inner binary in solar masses [%g]" %
  def_m0)
parser.add_option('-n', dest="m1", type = "float", default=def_m1,
  help = "Mass of second object in inner binary in solar masses [%g]" %
  def_m1)
parser.add_option('-o', dest="m2", type = "float", default=def_m2,
  help = "Mass of tertiary in solar masses [%g]" % def_m2)
parser.add_option('-O', dest="outfreq", type = "int", default=1,
  help = "Output frequency [%g]" % def_outfreq)
parser.add_option('-z', '--reo', dest="reo", type = "int", default=def_reo,
  help = "Reo [%g]" % def_reo)
parser.add_option('-p', '--phase', dest="phase", type = "float", default=def_phase,
  help = "Initial phase [%g]" % def_phase)
parser.add_option('-x', '--hex', dest="hex", type = "int", default=def_hex,
  help = "Hexadecapole [%g]" % def_hex)

options, remainder = parser.parse_args()

if options.phase > 2 * pi or (options.phase < 0 and options.phase != -1):
  print "Initial phase must be between 0 and 2pi or exactly -1 (for random)"
  sys.exit(1)
input_phase = options.phase

if input_phase == -1:
  phase0 = random.uniform(0, 2 * pi)
else:
  phase0 = input_phase

if options.reo != 0 and options.reo != 1:
  print "reos must be either on (1) or off (0)"
  sys.exit(1)
input_reo = options.reo

if options.hex != 0 and options.hex != 1:
  print "hexadecapole term must be either on (1) or off (0)"
  sys.exit(1)
input_hex = options.hex

class vec:
  '''A simple vector class.  This is to make the algebra in the Runge-Kutta
  easier.'''

  def __init__(self, x=7*[0]):
    if len(x) != 7:
      raise TypeError
    self.x = x

  def __getitem__(self, key):
    return self.x[key]

  def __repr__(self):
    return str(self.x)

  def __add__(self, v):
    newvec = []
    for elem1, elem2 in zip(self.x, v):
      newvec.append(elem1 + elem2)
    return vec(newvec)

  def __mul__(self, c):
    newvec = []
    for elem in self.x:
      newvec.append(elem * c)
    return vec(newvec)

  def __div__(self, c):
    newvec = []
    for elem in self.x:
      newvec.append(elem / c)
    return vec(newvec)

def rk4step(f, y, h, params):
  '''Take a 4th-order Runge-Kutta step.'''

  k1 = f(y, params[0], params[1]) * h
  k2 = f(y + k1 / 2., params[0], params[1]) * h
  k3 = f(y + k2 / 2., params[0], params[1]) * h
  k4 = f(y + k3, params[0], params[1]) * h

  return y + (k1 + k2 * 2 + k3 * 2 + k4) / 6.

def calc_H(e, i, consts, m):
  '''Calculates H.  See eq. 22 of Blaes et al. (2002)'''

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

def deriv(y, consts, m):
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

  C3 = (15 * G * m0 * m1 * m2 * (m0 - m1) / (64 * (m0 + m1)**2 * a2 * sqrt((1 -
    e2**2)**5)) * (a1 / a2)**3)
  
  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))

  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  th = (H**2 - G1**2 - G2**2) / (2 * G1 * G2)
  cosphi = - cosg1 * cosg2 - th * sing1 * sing2

  B = 2 + 5 * e1**2 - 7 * e1**2 * cos(2 * g1)

  A = 4 + 3 * e1**2 - 5 / 2. * (1 - th**2) * B

  da1dt = 0

  dg1dt = (6 * C2 * (1 / G1 * (4 * th**2 + (5 * cos(2 * g1) - 1) * (1 -
    e1**2 - th**2)) + th / G2 * (2 + e1**2 * (3 - 5 * cos(2 * g1)))) + C3 *
    e2 * e1 * (1 / G2 + th / G1) * (sing1 * sing2 * (A + 10 * (3 *
    th**2 - 1) * (1 - e1**2)) - 5 * th * B * cosphi) - C3 * e2 * (1 -
    e1**2) / (e1 * G1) * (10 * th * (1 - th**2) * (1 - 3 * e1**2) * sing1
    * sing2 + cosphi * (3 * A - 10 * th**2 + 2)))

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

  if input_reo:
    if e1 != 0:
      # See Eq. B10 from Ivanov (2005)
      phi = sqrt(G * (m0 + m1 + m2) / a2**3) * t + phase0
      alphabeta = (-1 / 2. * sin(2 * phi) * th * cos(2 * g1) +
        1 / 4. * sin(2 * g1) * (th**2 + cos(2 * phi) * (1 + th**2)))
      de1dt += (1 / sqrt(G * (m0 + m1) * a1) * sqrt(1 - e1**2) / e1 * 
        (15 / 2. * alphabeta * e1**2 * (a1 / a2)**3 * G * (m0 + m1) / a1))

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

  da2dt = 0

  dg2dt = (3 * C2 * (2 * th / G1 * (2 + e1**2 * (3 - 5 * cos(2 * g1))) + 1
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

  dHdt = 0

  return vec([da1dt, dg1dt, de1dt, da2dt, dg2dt, de2dt, dHdt])

consts = (6.67428e-11, 2.99792458e8)
solarmass = 1.9891e30
#consts = (4.49379e-15, 0.306389)

parsec = 3.0857e16

if options.m0 < 0:
  print "m0 must be greater than zero."
  sys.exit(1)
m0 = options.m0 * solarmass
if options.m1 < 0:
  print "m1 must be greater than zero."
  sys.exit(1)
m1 = options.m1 * solarmass
if options.m2 < 0:
  print "m2 must be greater than zero."
  sys.exit(1)
m2 = options.m2 * solarmass

if options.a1 < 0:
  print "a1 must be greater than zero."
  sys.exit(1)
a1 = options.a1 * parsec
if options.a2 < 0:
  print "a2 must be greater than zero."
  sys.exit(1)
a2 = options.a2 * parsec

if options.g1 < 0 or options.g1 >= 360:
  print "g1 must be between 0 and 360 degrees."
  sys.exit(1)
g1 = options.g1 * pi / 180.
if options.g2 < 0 or options.g2 >= 360:
  print "g2 must be between 0 and 360 degrees."
  sys.exit(1)
g2 = options.g2 * pi / 180.

if options.e1 < 0 or options.e1 > 1:
  print "e1 must be between 0 and 1."
  sys.exit(1)
e1 = options.e1
if options.e2 < 0 or options.e2 > 1:
  print "e2 must be between 0 and 1."
  sys.exit(1)
e2 = options.e2

if options.inc < -90 or options.inc > 90:
  print "inclination must be between -90 and 90 degrees."
  sys.exit(1)
inc = options.inc * pi / 180

if options.outfreq < 0 and options.outfreq != -1:
  print "output frequency must be greater than zero or exactly equal to -1"
  sys.exit(1)
outfreq = options.outfreq

m = (m0, m1, m2)
e = (e1, e2)

# a1 g1 e1 a2 g2 e2 H
yinit = [a1, g1, e1, a2, g2, e2, calc_H(e, inc, consts, m)]

y = vec(yinit)
print 0,
for elem in y:
  print elem,
print

endtime = options.endtime
timestep = options.timestep
t = 0
count = 0

# Calculate the Schwarzschild radius
schwarzschild_radius = 2 * consts[0] * max([m0, m1]) / consts[1]**2

print >> sys.stderr, "Old_BLS.py system parameters"
print >> sys.stderr
print >> sys.stderr, "a1 = ", a1, "a2 = ", a2
print >> sys.stderr, "m0 = ", m0, "m1 = ", m1, "m2 = ", m2
print >> sys.stderr, "e1 = ", e1, "e2 = ", e2
print >> sys.stderr, "g1 = ", g1, "g2 = ", g2
print >> sys.stderr, "inc = ", inc
print >> sys.stderr, "t_final = ", endtime, "tstep = ", timestep, "outfreq = ", outfreq
print >> sys.stderr, "reo = ", input_reo, "hex = ", input_hex, "phase = ", input_phase
print >> sys.stderr

print >> sys.stderr, yinit
print >> sys.stderr, endtime, timestep

# We seem to have a memory problem when we use odeint, so I'm just going to
# do 4th order Runge-Kutta.
#y = odeint(deriv, yinit, times, (consts, m, i))

while t < endtime:
  try:
    y = rk4step(deriv, y, timestep, (consts, m))
  except ValueError:
    print >> sys.stderr, "merge (valErr): ", t,
    for elem in y:
      print >> sys.stderr, elem,
    sys.exit()
    
  t += timestep
  if count != -1:
    if count % outfreq == 0:
      print t,
      for elem in y:
        print elem,
      print calc_cosi(consts, m, y)
    count += 1

  if y[0] < 10 * schwarzschild_radius:
    print >> sys.stderr, "merge: ", t,
    for elem in y:
      print >> sys.stderr, elem,
    sys.exit()

print >> sys.stderr, "timeout: ", t
