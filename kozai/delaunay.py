#! /usr/bin/env python 

'''
delaunay

Numerically integrate the dynamics of a hierarchical triple.
'''

# System modules
import json
import time

# Numerical modules
from math import sqrt, cos, sin, pi, acos
import numpy as np
from scipy.integrate import ode, quad

# Kozai modules
from _kozai_constants import *

class TripleDelaunay(object):
  '''Evolve a hierarchical triple using the Delaunay orbital elements (as
  opposed to the vectorial notation).  This class handles triples in which
  all objects are massive.  To integrate in the test particle approximation
  use the Triple_vector class.

  Parameters:
    a1: Semi-major axis of inner binary in AU
    a2: Semi-major axis of outer binary in AU
    e1: Eccentricity of inner binary
    e2: Eccentricity of outer binary
    inc: Inclination between inner and outer binaries in degrees
    g1: Argument of periapsis of the inner binary in degrees
    g2: Argument of periapsis of the outer binary in degrees
    m1: Mass of component 1 of the inner binary in solar masses
    m2: Mass of component 2 of the inner binary in solar masses
    m3: Mass of the tertiary in solar masses
    r1: Radius of component 1 of the inner binary in solar radii
    r2: Radius of component 2 of the inner binary in solar radii

  Other parameters:
    tstop: The time to integrate in years
    cputstop: The wall time to integrate in seconds
    outfreq: The number of steps between saving output
    atol: Absolute tolerance of the integrator
    rotl: Relative tolerance of the integrator
    quadrupole: Toggle the quadrupole term
    octupole: Toggle the octupole term
    hexadecapole: Toggle the hexadecapole term
    gr: Toggle GR effects
    algo: Set the integration algorithm (see the scipy.ode docs)
  '''

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, g1=0, g2=0, m1=1., 
    m2=1., m3=1., r1=0, r2=0):

    self._H = None

    self.a1 = a1
    self.a2 = a2
    self.e1 = e1
    self.e2 = e2
    self.g1 = g1
    self.g2 = g2
    self.m1 = m1
    self.m2 = m2
    self.m3 = m3
    self.r1 = r1
    self.r2 = r2
    self.inc = inc
    self.t = 0

    # Default integrator parameters
    self.tstop = None
    self.cputstop = 300
    self.outfreq = 1
    self.atol = 1e-9
    self.rtol = 1e-9
    self.quadrupole = True
    self.octupole  = True
    self.hexadecapole  = False
    self.gr = False
    self.algo = 'vode'
    self.maxoutput = 1e6
    self.collision = False

    # Store the initial state
    self.save_as_initial()

  ###
  ### Unit conversions & variable definitions
  ###
  ### Properties beginning with an underscore are stored in radians or SI
  ### units.  Most calculations are much easier when done in SI, but it is
  ### inconvenient for the user to deal with SI units.  Thus, the properties
  ### can be set using AU, M_sun, degrees, yr, or whatever other units are
  ### appropriate.
  ###

  # Times

  @property
  def t(self):
    '''Time in yr'''
    return self._t / yr2s

  @t.setter
  def t(self, val):
    '''Set time in yr'''
    self._t = val * yr2s
  
  # Masses

  @property
  def m1(self):
    '''m1 in solar masses'''
    return self._m1 / M_sun

  @m1.setter
  def m1(self, val):
    '''Set m1 in solar masses'''
    if self._H is not None:
      inc = self.inc

    self._m1 = val * M_sun

    if self._H is not None:
      self.inc = inc # Reset the total ang. momentum

  @property
  def m2(self):
    '''m2 in solar masses'''
    return self._m2 / M_sun

  @m2.setter
  def m2(self, val):
    '''Set m2 in solar masses'''
    if self._H is not None:
      inc = self.inc

    self._m2 = val * M_sun

    if self._H is not None:
      self.inc = inc # Reset the total ang. momentum

  @property
  def m3(self):
    '''m3 in solar masses'''
    return self._m3 / M_sun

  @m3.setter
  def m3(self, val):
    '''Set m3 in solar masses'''
    if self._H is not None:
      inc = self.inc

    self._m3 = val * M_sun

    if self._H is not None:
      self.inc = inc # Reset the total ang. momentum

  # Distances

  @property
  def a1(self):
    '''a1 in AU'''
    return self._a1 / au

  @a1.setter
  def a1(self, val):
    '''Set a1 in AU'''
    if self._H is not None:
      inc = self.inc

    self._a1 = val * au

    if self._H is not None:
      self.inc = inc # Reset the total ang. momentum

  @property
  def a2(self):
    '''a2 in AU'''
    return self._a2 / au

  @a2.setter
  def a2(self, val):
    '''Set a2 in AU'''
    if self._H is not None:
      inc = self.inc

    self._a2 = val * au

    if self._H is not None:
      self.inc = inc # Reset the total ang. momentum

  @property
  def r1(self):
    '''r1 in R_sun'''
    return self._r1 / R_sun

  @r1.setter
  def r1(self, val):
    '''r1 in R_sun'''
    self._r1 = val * R_sun

  @property
  def r2(self):
    '''r2 in R_sun'''
    return self._r2 / R_sun

  @r2.setter
  def r2(self, val):
    '''r2 in R_sun'''
    self._r2 = val * R_sun

  # Angles

  @property
  def g1(self):
    '''g1 in degrees'''
    return self._g1 * 180 / pi

  @g1.setter
  def g1(self, val):
    self._g1 = val * pi / 180

  @property
  def g2(self):
    '''g2 in degrees'''
    return self._g2 * 180 / pi

  @g2.setter
  def g2(self, val):
    self._g2 = val * pi / 180

  @property
  def cosphi(self):
    '''Angle between the arguments of periapsis.  See Eq. 23 of Blaes et al.
    (2002).'''
    return (-cos(self._g1) * cos(self._g2) - self.th * sin(self._g1) *
      sin(self._g2))
  
  @property
  def th(self):
    '''Calculate the cosine of the inclination.  See Eq. 22 of Blaes et al.
    (2002).'''
    return ((self._H**2 - self._G1**2 - self._G2**2) / (2 * self._G1 *
      self._G2))

  @property
  def _inc(self):
    '''The mutual inclination in radians.'''
    return acos(self.th)
  
  @property
  def inc(self):
    '''The mutual inclination.'''
    return self._inc * 180 / pi

  @inc.setter
  def inc(self, val):
    '''Set the inclination.  This really sets _H and inc is recalculated.'''
    self._H = sqrt(self._G1**2 + self._G2**2 + 2 * self._G1 * self._G2 *
      cos(val * pi / 180))

  # Angular momenta

  @property
  def _G1(self):
    '''Calculate G1.  See Eq. 6 of Blaes et al. (2002).'''
    return (self._m1 * self._m2 * sqrt(G * self._a1 * (1 -
      self.e1**2) / (self._m1 + self._m2)))

  @property
  def _G2(self):
    '''Calculate G2.  See Eq. 7 of Blaes et al. (2002).'''
    return ((self._m1 + self._m2) * self._m3 * sqrt(G * self._a2 * (1
      - self.e2**2) / (self._m1 + self._m2 + self._m3)))

  # Energies

  @property
  def C2(self):
    '''Calculate C2.  See Eq. 18 of Blaes et al., (2002).'''
    return (G * self._m1 * self._m2 * self._m3 / (16 * (self._m1 +
      self._m2) * self._a2 * (1 - self.e2**2)**(3./2)) * (self._a1 /
      self._a2)**2)

  @property
  def C3(self):
    '''Calculate C3.  See Eq. 19 of Blaes et al., (2002).'''
    return (15 * G * self._m1 * self._m2 * self._m3 * (self._m1 -
      self._m2) / (64 * (self._m1 + self._m2)**2 * self._a2 * (1 -
      self.e2**2)**(5./2)) * (self._a1 / self._a2)**3)

  # Other parameters

  @property
  def epsoct(self):
    return self.e2 / (1 - self.e2**2) * (self.a1 / self.a2)

  @property
  def Th(self):
    '''Calculate Kozai's integral.'''
    return (1 - self.e1**2) * cos(self._inc)**2

  @property
  def CKL(self):
    '''Calculate the libration constant.'''
    return self.e1**2 * (1 - 5./2 * sin(self._inc)**2 * sin(self._g1)**2)

  @property
  def Hhatquad(self):
    '''The normalized quadrupole term of the Hamiltonian'''
    return ((2 + 3 * self.e1**2) * (1 - 3 * self.th**2) - 15 *
      self.e1**2 * (1 - self.th**2) * cos(2 * self._g1))

  @property
  def outfreq(self):
    return self._outfreq

  @outfreq.setter
  def outfreq(self, val):
    self._outfreq = int(val)

  def save_as_initial(self):
    '''Set the current parameters as the initial parameters.'''

    self.initial_state = {}
    self.initial_state['a1'] = self.a1
    self.initial_state['a2'] = self.a2
    self.initial_state['e1'] = self.e1
    self.initial_state['e2'] = self.e2
    self.initial_state['g1'] = self.g1
    self.initial_state['g2'] = self.g2
    self.initial_state['m1'] = self.m1
    self.initial_state['m2'] = self.m2
    self.initial_state['m3'] = self.m3
    self.initial_state['r1'] = self.r1
    self.initial_state['r2'] = self.r2
    self.initial_state['inc'] = self.inc

  ###
  ### Integration routines
  ###

  def _deriv(self, t, y):
    '''The EOMs.  See Eqs. 11 -- 17 of Blaes et al. (2002).'''

    # Unpack the values
    a1, e1, g1, e2, g2, H = y

    # Calculate trig functions only once
    sing1 = sin(g1)
    sing2 = sin(g2)
    cosg1 = cos(g1)
    cosg2 = cos(g2)

    m1 = self._m1
    m2 = self._m2
    m3 = self._m3
    a2 = self._a2
  
    # TODO
    # Are these necessary now that we are calculating them dynamically?
    G1 = m1 * m2 * sqrt(G * a1 * (1 - e1**2) / (m1 + m2))
    G2 = (m1 + m2) * m3 * sqrt(G * a2 * (1 - e2**2) / (m1 + m2 + m3))

    C2 = (G * m1 * m2 * m3 / (16 * (m1 + m2) * a2 * (1 - e2**2)**(3./2)) * 
          (a1 / a2)**2)
    C3 = (15 * G * m1 * m2 * m3 * (m2 - m1) / (64 * (m1 + m2)**2 * a2 *
          (1 - e2**2)**(5./2)) * (a1 / a2)**3)

    th = (H**2 - G1**2 - G2**2) / (2 * G1 * G2)
    cosphi = cosg1 * cosg2 - th * sing1 * sing2
    B = 2 + 5 * e1**2 - 7 * e1**2 * cos(2 * g1)
    A = 4 + 3 * e1**2 - 5 / 2. * (1 - th**2) * B

    # Eq. 11 of Blaes et al. (2002)
    da1dt = 0.
    if self.gr:
      da1dt += -(64 * G**3 * m1 * m2 * (m1 + m2) / (5 * c**5 * a1**3 * 
        sqrt((1 - e1**2)**7)) * (1 + 73 / 24. * e1**2 + 37 / 96. * e1**4))

    # Eq. 12 of Blaes et al. (2002)
    dg1dt = 0.
    if self.quadrupole:
      dg1dt += (6 * C2 * (1 / G1 * (4 * th**2 + (5 * cos(2 * g1) - 1) * (1 -
        e1**2 - th**2)) + th / G2 * (2 + e1**2 * (3 - 5 * cos(2 * g1)))))
    if self.octupole:
      dg1dt += (C3 * e2 * e1 * (1 / G2 + th / G1) * (sing1 * sing2 * 
        (A + 10 * (3 * th**2 - 1) * (1 - e1**2)) - 5 * th * B * cosphi) - C3
        * e2 * (1 - e1**2) / (e1 * G1) * (10 * th * (1 - th**2) * (1 - 3 *
        e1**2) * sing1 * sing2 + cosphi * (3 * A - 10 * th**2 + 2)))
    if self.gr:
      dg1dt += ((3 / (c**2 * a1 * (1 - e1**2)) * 
        sqrt((G * (m1 + m2) / a1)**3)))
    if self.hexadecapole:
      dg1dt += (1 / (4096. * a2**5 * sqrt(1 - e1**2) * (m1 + m2)**5) * 45 *
        a1**3 * sqrt(a1 * G * (m1 + m2)) * (-1 / ((e2**2 - 1)**4 * sqrt(a2 *
        G * (m1 + m2 + m3))) * (m1**2 - m1 * m2 + m2**2) * (sqrt(1 - e2**2) *
        m2**2 * m3 * sqrt(a2 * G * (m1 + m2 + m3)) * th + m1**2 * (sqrt( 1 -
        e1**2) * m2 * sqrt(a1 * G * (m1 + m2)) + sqrt(1 - e2**2) * m3 *
        sqrt(a2 * G * (m1 + m2 + m3)) * th) + m1 * m2 * (sqrt(1 - e1**2) * m2
        * sqrt(a1 * G * (m1 + m2)) + sqrt(1 - e1**2) * sqrt(a1 * G * (m1
        + m2)) * m3 + 2 * sqrt(1 - e2**2) * m3 * sqrt(a2 * G * (m1 + m2 +
        m3)) * th)) * (96 * th + 480 * e1**2 * th + 180 * e1**4 * th + 144 *
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
        2 * (1 - e1**2) * (m1 + m2) * (m1**3 + m2**3) * m3 * (e1 * (4 + 3 *
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
    de1dt = 0.
    if self.quadrupole:
      de1dt += (30 * C2 * e1 * (1 - e1**2) / G1 * (1 - th**2) * sin(2 * g1))
    if self.octupole:
      de1dt += (-C3 * e2 * (1 - e1**2) / G1 * (35 * cosphi * (1 - th**2) * 
        e1**2 * sin(2 * g1) - 10 * th * (1 - e1**2) * (1 - th**2) * 
        cosg1 * sing2 - A * (sing1 * cosg2 - th * cosg1 * sing2)))
    if self.gr:
      de1dt += (-304 * G**3 * m1 * m2 * (m1 + m2) * e1 / (15 * c**4 * a1**4 * 
        sqrt((1 - e1**2)**5)) * (1 + 121 / 304. * e1**2))
    if self.hexadecapole:
      de1dt += (-(315 * a1**3 * e1 * sqrt(1 - e1**2) * sqrt(a1 * G * (m1 +
      m2)) * (m1**2 - m1 * m2 + m2**2) * m3 * (2 * (2 + e1**2) * (2 + 3 *
      e2**2) * (1 - 8 * th**2 + 7 * th**4) * sin(2 * g1) - 21 * e1**2 * (2 +
      3 * e2**2) * (th**2 - 1)**2 * sin(4 * g1) + e2**2 * (21 * e1**2 * (th -
      1) * (1 + th)**3 * sin(4 * g1 - 2 * g2) - 2 * (2 + e1**2) * (1 + th)**2
      * (1 - 7 * th + 7 * th**2) * sin(2 * (g1 - g2)) - (th - 1)**2 * (2 * (2
      + e1**2) * (1 + 7 * th + 7 * th**2) * sin(2 * (g1 + g2)) - 21 * e1**2 *
      (th**2 - 1) * sin(2 * (2 * g1 + g2)))))) / (2048 * a2**5 * sqrt((1 -
      e2**2)**7) * (m1 + m2)**3))

    dg2dt = 0.
    if self.quadrupole:
      dg2dt += (3 * C2 * (2 * th / G1 * (2 + e1**2 * (3 - 5 * cos(2 * g1))) + 1
        / G2 * (4 + 6 * e1**2 + (5 * th**2 - 3) * (2 + 3 * e1**2 - 5 * e1**2 *
        cos(2 * g1))))) 
    if self.octupole:
      dg2dt += (-C3 * e1 * sing1 * sing2 * ((4 * e2**2 + 1) / (e2 * G2) * 10 * 
        th * (1 - th**2) * (1 - e1**2) - e2 * (1 / G1 + th / G2) * (A + 10 * 
        (3 * th**2 - 1) * (1 - e1**2))) - C3 * e1 * cosphi * (5 * B * th * 
        e2 * (1 / G1 + th / G2) + (4 * e2**2 + 1) / (e2 * G2) * A))
    if self.hexadecapole:
      dg2dt += ((9 * a1**3 * (-1 / sqrt(1 - e1**2) * 10 * a2 * sqrt(a1 * G *
      (m1 + m2)) * (m1**2 - m1 * m2 + m2**2) * (sqrt(1 - e2**2) * m2**2 * m3
      * sqrt(a2 * G * (m1 + m2 + m3)) + m1**2 * (sqrt(1 - e2**2) * m3 *
      sqrt(a2 * G * (m1 + m2 + m3)) + sqrt(1 - e1**2) * m2 * sqrt(a1 * G *
      (m1 + m2)) * th) + m1 * m2 * (2 * sqrt(1 - e2**2) * m3 * sqrt(a2 * G *
      (m1 + m2 + m3)) + sqrt(1 - e1**2) * m2 * sqrt(a1 * G * (m1 + m2)) * th
      + sqrt(1 - e1**2) * sqrt(a1 * G * (m1 + m2)) * m3 * th)) * (96 * th +
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
      e1**4 * e2**2 * th**3 * cos(2 * (2 * g1 + g2))) + a1 * a2 * G * m1 * m2
      * (m1**3 + m2**3) * (m1 + m2 + m3) * (-6 * (8 + 40 * e1**2 + 15 *
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
      + g2)))))) / (8192 * a2**6 * (-1 + e2**2)**4 * (m1 + m2)**5 * sqrt(a2 *
      G * (m1 + m2 + m3))))

    # Eq. 16 of Blaes et al. (2002)
    de2dt = 0.
    if self.octupole:
      de2dt += (C3 * e1 * (1 - e2**2) / G2 * (10 * th * (1 - th**2) * (1 -
      e1**2) * sing1 * cosg2 + A * (cosg1 * sing2 - th * sing1 * cosg2)))
    if self.hexadecapole:
      de2dt += ((45 * a1**4 * e2 * m1 * m2 * (m1**2 - m1 * m2 + m2**2) *
        sqrt(a2 * G * (m1 + m2 + m3)) * (-147 * e1**4 * (-1 + th) * (1 +
        th)**3 * sin(4 * g1 - 2 * g2) + 28 * e1**2 * (2 + e1**2) * (1 +
        th)**2 * (1 - 7 * th + 7 * th**2) * sin(2 * (g1 - g2)) + (-1 + th) *
        (2 * (8 + 40 * e1**2 + 15 * e1**4) * (-1 - th + 7 * th**2 + 7 *
        th**3) * sin(2 * g2) - 7 * e1**2 * (-1 + th) * (4 * (2 + e1**2) * (1
        + 7 * th + 7 * th**2) * sin(2 * (g1 + g2)) - 21 * e1**2 * (-1 + th**2)
        * sin(2 * (2 * g1 + g2))))) / (4096 * a2**6 * (-1 + e2**2)**3 * (m1 +
        m2)**4)))

    # Eq. 17 of Blaes et al. (2002)
    dHdt = 0.
    if self.gr:
      dHdt += (-32 * G**3 * m1**2 * m2**2 / (5 * c**5 * a1**3 * 
        (1 - e1**2)**2) * sqrt(G * (m1 + m2) / a1) * (1 + 7 / 8. * e1**2) * 
        (G1 + G2 * th) / H)

    der = [da1dt, de1dt, dg1dt, de2dt, dg2dt, dHdt]
    return der

  def _step(self):
    self.solver.integrate(self.tstop, step=True)
    self.nstep += 1
    self._t = self.solver.t
    self._a1, self.e1, self._g1, self.e2, self._g2, self._H = self.solver.y
    self._g1 %= (2 * pi)
    self._g2 %= (2 * pi)

  def integrator_setup(self):
    '''Set up the integrator.'''

    # Integration parameters
    self.nstep = 0

    self._y = [self._a1, self.e1, self._g1, self.e2, self._g2, self._H]

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.algo, nsteps=1, atol=self.atol, 
      rtol=self.rtol)
    self.solver.set_initial_value(self._y, self._t)
    if self.algo == 'vode':
      self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def reset(self):
    '''Reset the triple to its initial configuration.  This resets the
    orbital parameters and time, but does not reset the integration
    options.'''
    self.t = 0
    for key in self.initial_state:
      setattr(self, key, self.initial_state[key])

  def evolve(self, tstop):
    '''Integrate the triple in time.

    Parameters:
      tstop: The time to integrate in years
    '''
    
    self.tstop = tstop
    n_columns = len(self.state())
    self.integrator_setup()
    self.integration_steps = np.zeros((self.maxoutput, n_columns))
    self.integration_steps[0] = self.state()

    self.tstart = time.time()
    while ((self.t < tstop) and 
      ((time.time() - self.tstart) < self.cputstop)):

      self._step()
      if self.nstep % self.outfreq == 0:
        self.integration_steps[self.nstep/self.outfreq] = self.state()

      if self.a1 * (1 - self.e1) < self.r1 + self.r2:
        self.collision = True
        break

    laststep = (self.nstep / self.outfreq) + 1
    self.integration_steps[laststep] = self.state()

    return self.integration_steps[:laststep+1]

  def extrema(self, tstop):
    '''Integrate the triple, but only save the eccentricity extrema.
    
    Parameters:
      tstop: The time to integrate in years
    '''

    self.tstop = tstop
    n_columns = len(self.state())
    self.integrator_setup()
    self.integration_steps = np.zeros((self.maxoutput, n_columns))

    t_prev = 0
    e_prev = 0
    e_prev2 = 0
    output_index = 0
    self.tstart = time.time()

    while (self.t < self.tstop and 
      time.time() - self.tstart < self.cputstop):

      self._step()
      if e_prev2 < e_prev > self.e1:
        self.integration_steps[output_index] = prevstate
        output_index += 1
      elif e_prev2 > e_prev < self.e1:
        self.integration_steps[output_index] = prevstate
        output_index += 1

      # Check for collisions
      if self.a1 * (1 - self.e1) < self.r1 + self.r2:
        self.collision = True
        break

      t_prev = self.t
      e_prev2 = e_prev
      e_prev = self.e1
      prevstate = self.state()

    return self.integration_steps[:output_index]

  def find_flips(self, tstop):
    '''Integrate the triple, but print out only when there is a flip.'''

    self.tstop = tstop
    n_columns = len(self.state())
    self.integrator_setup()
    self.integration_steps = np.zeros((self.maxoutput, n_columns))

    t_prev = 0
    e_prev = 0
    e_prev2 = 0
    sign_prev = np.sign(self.th)
    output_index = 0
    self.tstart = time.time()
    while (self.t < self.tstop and 
      time.time() - self.tstart < self.cputstop):
      self._step()
      if e_prev2 < e_prev > self.e1:
        if np.sign(self.th) != sign_prev:
          self.integration_steps[output_index] = prevstate
          output_index += 1
        sign_prev = np.sign(self.th)
      t_prev = self.t
      e_prev2 = e_prev
      e_prev = self.e1
      prevstate = self.state()

    return self.integration_steps[:output_index]

  def state(self):
    '''Return a tuple with the dynamical state of the system.

    Returns:
      (t, a1, e1, g1, a2, e2, g2, inc)
    '''
    return (self.t, self.a1, self.e1, self.g1, self.a2, self.e2, self.g2, 
      self.inc)
  
  def __repr__(self):
    '''Print out the initial values in JSON format.'''

    # Get the initial state
    json_data = self.initial_state

    # Add some other properties
    json_data['epsoct'] = self.epsoct
    json_data['tstop'] = self.tstop
    json_data['cputstop'] = self.cputstop
    json_data['outfreq'] = self.outfreq
    json_data['atol'] = self.atol
    json_data['rtol'] = self.rtol
    json_data['quadrupole'] = self.quadrupole
    json_data['octupole'] = self.octupole
    json_data['hexadecapole'] = self.hexadecapole
    json_data['gr'] = self.gr
    json_data['algo'] = self.algo
    json_data['maxoutput'] = self.maxoutput
    json_data['collision'] = self.collision

    return json.dumps(json_data, sort_keys=True, indent=2)
