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

from math import sqrt, cos, sin, pi, acos
from numpy import linspace
import sys
import optparse
import pdb
import pygsl
import pygsl._numobj as Numeric
import time
from pygsl import odeiv
import random

#pygsl.set_debug_level(2) 

#####  constants  (made to match Fewbody, though in mks) ####
consts = (6.67384e-11, 2.99792458e8)
solarmass = 1.989e30
parsec = 3.0857e16
au = 1.496e11
year = 3.155693e7
Rsun = 6.9599e8
########


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

parser = optparse.OptionParser()
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
parser.add_option('-m', '--m00', dest="m0", type = "float", default=def_m0,
  help = "Mass of first object in inner binary in solar masses [%g]" %
  def_m0)
parser.add_option('-n', '--m01', dest="m1", type = "float", default=def_m1,
  help = "Mass of second object in inner binary in solar masses [%g]" %
  def_m1)
parser.add_option('-o', '--m1', dest="m2", type = "float", default=def_m2,
  help = "Mass of tertiary in solar masses [%g]" % def_m2)
parser.add_option('-F', '--freq', dest="outfreq", type = "int", default=def_outfreq,
  help = "Output frequency [%g]" % def_outfreq)
parser.add_option('-r', '--aacc', dest="absacc", type = "float", default=def_absacc,
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

options, remainder = parser.parse_args()

if options.m0 < 0:
  print >> sys.stderr, "m0 must be greater than zero."
  sys.exit(1)
input_m0 = options.m0 * solarmass
if options.m1 < 0:
  print >> sys.stderr, "m1 must be greater than zero."
  sys.exit(1)
input_m1 = options.m1 * solarmass
if options.m2 < 0:
  print >> sys.stderr, "m2 must be greater than zero."
  sys.exit(1)
input_m2 = options.m2 * solarmass
if options.a1 < 0:
  print >> sys.stderr, "a1 must be greater than zero."
  sys.exit(1)
input_a1 = options.a1 * au
if options.a2 < 0:
  print >> sys.stderr, "a2 must be greater than zero."
  sys.exit(1)
input_a2 = options.a2 * au
if options.g1 < 0 or options.g1 >= 360:
  print >> sys.stderr, "g1 must be between 0 and 360."
  sys.exit(1)
input_g1 = options.g1 * pi / 180.
if options.g2 < 0 or options.g2 >= 360:
  print >> sys.stderr, "g2 must be between 0 and 360."
  sys.exit(1)
input_g2 = options.g2 * pi / 180.
if options.e1 < 0 or options.e1 > 1:
  print >> sys.stderr, "e1 must be between 0 and 1."
  sys.exit(1)
input_e1 = options.e1
if options.e2 < 0 or options.e2 > 1:
  print >> sys.stderr, "e2 must be between 0 and 1."
  sys.exit(1)
input_e2 = options.e2
if options.relacc > 1 or options.relacc < 0:
  print >> sys.stderr, "relacc must be between 0 and 1."
  sys.exit(1)
input_relacc = options.relacc
if options.absacc < 0 or options.absacc > 1:
  print >> sys.stderr, "absacc must be between 0 and 1."
  sys.exit(1)
input_absacc = options.absacc
if options.cputime < 0 and options.cputime != -1:
  print >> sys.stderr, "cputime must greater than 0 of exactly -1."
  sys.exit(1)
input_cputime = options.cputime
if options.inc < -90 or options.inc > 90:
  print >> sys.stderr, "cosi must be between -90 and 90 degrees."
  sys.exit(1)
input_inc = options.inc * pi / 180.
if options.outfreq < 0 and options.outfreq != -1:
  print >> sys.stderr, "output frequency must be greater than zero or exactly equal to -1"
  sys.exit(1)
input_outfreq = options.outfreq
if options.verb != 0 and options.verb != 1:
  print >> sys.stderr, "verbose must either be on (1) or off (0)"
  sys.exit(1)
input_verb = options.verb
if options.oct != 0 and options.oct != 1:
  print >> sys.stderr, "octupole terms must either be on (1) or off (0)"
  sys.exit(1)
input_oct = options.oct
if options.gr != 0 and options.gr != 1:
  print >> sys.stderr, "general relativity must either be on (1) or off (0)"
  sys.exit(1)
input_gr = options.gr
if options.reo != 0 and options.reo != 1:
  print >> sys.stderr, "reos must be either on (1) or off (0)"
  sys.exit(1)
input_reo = options.reo
if options.phase > 2 * pi or (options.phase < 0 and options.phase != -1):
  print >> sys.stderr, "Initial phase must be between 0 and 2pi or exactly -1 (for random)"
  sys.exit(1)
input_phase = options.phase
if options.hex != 0 and options.hex != 1:
  print >> sys.stderr, "hexadecapole term must be either on (1) or off (0)"
  sys.exit(1)
input_hex = options.hex


if input_phase == -1:
  phase0 = random.uniform(0, 2 * pi)
else:
  phase0 = input_phase

if input_cputime == -1:
    input_cputime = float('inf')

m = (input_m0, input_m1, input_m2)




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
      C3 = (15 * G * m0 * m1 * m2 * (m0 - m1) / (64 * (m0 + m1)**2 * a2 * sqrt((1 -
        e2**2)**5)) * (a1 / a2)**3)
  
  G1 = m0 * m1 * sqrt(G * a1 * (1 - e1**2) / (m0 + m1))

  G2 = (m0 + m1) * m2 * sqrt(G * a2 * (1 - e2**2) / (m0 + m1 + m2))

  th = (H**2 - G1**2 - G2**2) / (2 * G1 * G2)
  cosphi = - cosg1 * cosg2 - th * sing1 * sing2

  B = 2 + 5 * e1**2 - 7 * e1**2 * cos(2 * g1)

  A = 4 + 3 * e1**2 - 5 / 2. * (1 - th**2) * B


  
  da1dt = 0.0
  if input_gr:
      da1dt += - (64 * G**3 * m0 * m1 * (m0 + m1) / (5 * c**5 * a1**3 * sqrt((1 -    e1**2)**7)) * (1 + 73 / 24. * e1**2 + 37 / 96. * e1**4))

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
    dHdt += (-32 * G**3 * m0**2 * m1**2 / (5 * c*c*c*c*c * a1**3 * (1 - e1**2)**2) * sqrt(G * (m0 + m1) / a1) * (1 + 7 / 8. * e1**2) * (G1 + G2 * th) / H)

  der = (da1dt, dg1dt, de1dt, da2dt, dg2dt, de2dt, dHdt)
  #pdb.set_trace()
  return der




#make the global triple variables local
m0 = input_m0
m1 = input_m1
m2 = input_m2
a1 = input_a1
a2 = input_a2
g1 = input_g1
g2 = input_g2
e1 = input_e1
e2 = input_e2
relacc = input_relacc
absacc = input_absacc
inc = input_inc
outfreq = input_outfreq

dimension = 7
# The different possible steppers for the function
# Comment your favorite one out to test it.
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


m = (m0, m1, m2)
e = (e1, e2)
a = (a1, a2)
# 0  1  2  3  4  5  6
# a1 g1 e1 a2 g2 e2 H
yinit = (a1, g1, e1, a2, g2, e2, calc_H(e, inc, consts, m, a))

endtime = options.endtime * year
cputime = input_cputime

# Calculate the Schwarzschild radius
r_schw = 2 * consts[0] * max([m0, m1]) / consts[1]**2
merger = .01 * a1

#initial step size h, it will be adapted by apply
time_step = 1000.0 * year
tinit = 0.0

print >> sys.stderr, "BLS.py system parameters"
print >> sys.stderr
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

t = tinit
y = yinit
stamp = time.time()
count = 0
print t/year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], calc_cosi(consts, m, y)   
while (t < endtime):
  if ((time.time() - stamp) >  cputime): #stop if cpulimit is exceeded
    #print of the final state
    if outfreq != -1:
      print t/year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], calc_cosi(consts, m, y)   
    
    raise ValueError, "cpu limit reached!"
    break

  t, time_step, y = evolve.apply(t, endtime, time_step, y)
  y[1] = y[1] % (2.0 * pi)
  y[4] = y[4] % (2.0 * pi)
  
  if outfreq != -1:
    if (count % outfreq == 0):
      print t/year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], calc_cosi(consts, m, y)
  count += 1
          
  if (y[0] < merger):
    print t/year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], calc_cosi(consts, m, y)
    print >> sys.stderr, "orbit is 1% of its original value -- merger: "
    break
  elif (y[0] * (1 - y[2]) < 10 * r_schw):
    print t/year, y[0]/au, y[1], y[2], y[3]/au, y[4], y[5], y[6], calc_cosi(consts, m, y)
    print >> sys.stderr, "collision -- merger: "
    break

if input_verb == 1:
  print >> sys.stderr, "# Using stepper %s with order %d" %(step.name(), step.order())
  print >> sys.stderr, "# Using Control ", control.name()
  print >> sys.stderr, yinit
  print >> sys.stderr, endtime, time_step 
  print >> sys.stderr, "Needed %f seconds" %( time.time() - stamp,)
    
