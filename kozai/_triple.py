#! /usr/bin/env python

'''
_triple

The base class for triples.
'''

class Triple(object):
  '''
  A base class for triples.  Contains the dynamical and physical
  characteristics of the system (orbital elements, mass, etc.)

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
  '''

  def __init__(self, a1=1, a2=10, e1=.1, e2=.1, inc=80, g1=0, m1=1, m3=1,
    r1=0, param_file=None):

    self.a1 = float(a1)
    self.a2 = float(a2)
    self.e1 = float(e1)
    self.e2 = float(e2)
    self.inc = inc * np.pi / 180
    self.g1 = g1 * np.pi / 180
    self.m1 = float(m1)
    self.m2 = float(m2)
    self.m3 = float(m3)
    self.r1 = float(r1)
    self.r2 = float(r2)
