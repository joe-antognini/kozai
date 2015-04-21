#! /usr/bin/env python

'''
ts_vector

Numerically integrate the dynamics of a hierarchical triple in the test
particle limit using vectorial notation.
'''

# System packages
import sys
import time

# Numerical packages
from math import pi, sin, cos, acos
import numpy as np
from scipy.integrate import ode, quad
from scipy.optimize import root, fsolve

# Kozai modules
from _kozai_constants import *

class TripleVectorial(object):
  '''Evolve a triple in time using the vectorial equations of motion.  This
  class only applies to a triple in the test particle approximation.  For
  triples with a massive secondary, use the TripleDelaunay class.
  
  Parameters:
    a1: Semi-major axis of inner binary in AU
    a2: Semi-major axis of outer binary in AU
    e1: Eccentricity of inner binary
    e2: Eccentricity of outer binary
    inc: Inclination between inner and outer binaries in degrees
    g1: Argument of periapsis of the inner binary in degrees
    longascnode: Longitude of ascending node in degrees
    m1: Mass of component 1 of the inner binary in solar masses
    m3: Mass of the tertiary in solar masses
    tstop: The time to integrate in years

  Other parameters:
    tstop: The time to integrate in years
    cputstop: The wall time to integrate in seconds
    outfreq: The number of steps between saving output
    atol: Absolute tolerance of the integrator
    rotl: Relative tolerance of the integrator
    quadrupole: Toggle the quadrupole term
    octupole: Toggle the octupole term
    algo: Set the integration algorithm (see the scipy.ode docs)
  '''

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, g1=0, m1=1, m3=1,
    longascnode=180.):

    # First set the vectorial elements
    self.jhatvec = np.array([
      sin(inc) * sin(Omega),
      -sin(inc) * cos(Omega),
      cos(inc)])
    self.jvec = sqrt(1 - e1**2) * self.jhatvec

    ehatvec_sol = root(_evec_root, [.5, .5, .5], (self.jhatvec, g1 * 
      pi / 180))
    self.ehatvec = ehatvec_sol.x
    self.evec = e1 * self.ehatvec

    self.a1 = a1
    self.a2 = a2
    self.e1 = e1
    self.e2 = e2
    self.g1 = g1
    self.Omega = longascnode
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
    self.algo = 'vode'
    self.maxoutput = 1e6
    self.collision = False

    # Store the initial state
    self.save_as_initial()

    self.update()

    # Integration parameters
    self.nstep = 0
    self.tstop = tstop
    self.cputstop = cputstop
    self.outfreq = outfreq
    self.outfilename = outfilename
    self.integration_algo = integration_algo
    self.y = list(np.concatenate((self.jvec, self.evec)))

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
    return self._t * self.tsec / yr2s

  @t.setter
  def t(self, val):
    '''Set the time in yr'''
    self._t = val * yr2s / self.tsec

  @property
  def tsec(self):
    '''The secular timescale'''
    return sqrt(G * self._m1 * self._a1) / self.Phi0

  # Masses

  @property
  def m1(self):
    '''m1 in solar masses'''
    return self._m1 / M_sun

  @m1.setter
  def m1(self, val):
    '''Set m1 in solar masses'''
    self._m1 = val * M_sun

  @property
  def m3(self):
    '''m3 in solar masses'''
    return self._m3 / M_sun

  @m3.setter
  def m3(self, val):
    '''Set m3 in solar masses'''
    self._m3 = val * M_sun

  # Distances

  @property
  def a1(self):
    '''a1 in AU'''
    return self._a1 / au

  @a1.setter
  def a1(self, val):
    '''Set a1 in AU'''
    self._a1 = val * au

  @property
  def a2(self):
    '''a2 in AU'''
    return self._a2 / au

  @a2.setter
  def a2(self, val):
    '''Set a2 in AU'''
    self._a2 = val * au

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
  def th(self):
    '''Cosine of the inclination'''
    return cos(self._inc)

  @property
  def _inc(self):
    '''The mutual inclination in radians'''
    return acos(self.jhatvec[2])

  @property
  def inc(self):
    '''The mutual inclination in degrees'''
    return self._inc * 180 / pi

  # Other parameters

  @property
  def e1(self):
    '''The eccentricity of the inner binary'''
    return np.linalg.norm(self.evec)
  
  @property
  def j(self):
    '''The normalized angular momentum of the inner binary'''
    return sqrt(1 - self.e1**2)

  @property
  def Phi0(self):
    '''The normalization of the potential.'''
    return (G * self._m3 * self._a1**2 / (self._a2**3 * (1 -
      self.e2**2)**(3./2)))

  @property
  def epsoct(self):
    '''The strength of the octupole term relative to the quadrupole'''
    return self.e2 / (1 - self.e2**2) * (self.a1 / self.a2)

  @property
  def Hhatquad(self):
    '''The normalized quadrupole term of the Hamiltonian'''
    return ((2 + 3 * self.e1**2) * (1 - 3 * self.th**2) - 15 *
      self.e1**2 * (1 - self.th**2) * cos(2 * self._g1))

  @property
  def phiq(self):
    '''The quadrupole term of the potential'''
    return (3/4. * (self.jvec[2]**2 / 2. + self.e1**2 - 5/2. * 
      self.evec[2]**2 - 1/6.)

  @property
  def phioct(self):
    '''The octupole term of the potential'''
      return (self.epsoct * 75/64. * (self.evec[0] * (1/5. - 8/5. *
        self.e1**2 + 7 * self.evec[2]**2 - self.jvec[2]**2) - 2 *
        self.evec[2] * self.jvec[0] * self.jvec[2]))

  @property
  def Th(self):
    '''Calculate Kozai's integral.'''
    return (1 - self.e1**2) * cos(self._inc)**2

  @property
  def CKL(self):
    '''Calculate the libration constant.'''
    return self.e1**2 * (1 - 5./2 * sin(self._inc)**2 * sin(self._g1)**2)

  def _save_initial_params(self):
    '''Set the variables to their initial values.  Just a clone of
    reset().'''

    # Ordinary variables
    self.e1_0 = self.e1
    self.e2_0 = self.e2
    self.inc_0 = self.inc
    self.Omega_0 = self.Omega
    self.g1_0 = self.g1
    self.j_0 = self.j
    self.nstep = 0

    # Arrays need to be deep copied
    self.jhatvec_0 = self.jhatvec[:]
    self.jvec_0 = self.jvec[:]
    self.ehatvec_0 = self.ehatvec[:]
    self.evec_0 = self.evec[:]

  def reset(self):
    '''Set the variables to their initial values.'''

    # Ordinary variables
    self.e1 = self.e1_0
    self.e2 = self.e2_0
    self.inc = self.inc_0
    self.Omega = self.Omega_0
    self.g1 = self.g1_0
    self.j = self.j_0
    self.nstep = 0
    self._t = 0

    # Arrays need to be deep copied
    self.jhatvec = self.jhatvec_0[:]
    self.jvec = self.jvec_0[:]
    self.ehatvec = self.ehatvec_0[:]
    self.evec = self.evec_0[:]

    self.update()

  def update(self):
    '''Update the derived parameters.'''
    self.calc_Th()
    self.calc_CKL()
    self.t = self._t * self.tsec
    self.e1 = np.linalg.norm(self.evec)
  
  def _deriv(self, t, y, epsoct):
    '''The EOMs.  See Eqs. 4 of Katz et al. (2011).'''

    # Note that we have the following correspondences:
    # y[0]  y[1]  y[2]  y[3]  y[4]  y[5]
    # j_x   j_y   j_z   e_x   e_y   e_z

    #The total eccentricity:
    jx, jy, jz, ex, ey, ez = y
    e_sq = ex**2 + ey**2 + ez**2

    # Calculate the derivatives of phi.
    grad_j_phi_q = np.array([0, 0, 3/4. * jz])
    grad_j_phi_oct = -75/32. * np.array([ez * jz, 0,
      ex * jz + ez * jx])
    grad_e_phi_q = np.array([3/2. * ex, 3/2. * ey, -9/4. * ez])
    grad_e_phi_oct = np.array([
      75/64. * (1/5. - 8/5. * e_sq + 7 * ez**2 - jz**2) - 15/4. * ex**2,
      -15/4. * ex * ey,
      75/64. * (54/5. * ex * ez - 2 * jx * jz)])

    grad_j_phi = grad_j_phi_q + epsoct * grad_j_phi_oct
    grad_e_phi = grad_e_phi_q + epsoct * grad_e_phi_oct

    djdtau = np.cross(y[:3], grad_j_phi) + np.cross(y[3:], grad_e_phi)
    dedtau = np.cross(y[:3], grad_e_phi) + np.cross(y[3:], grad_j_phi)

    ret = np.concatenate((djdtau, dedtau))
    return list(ret)

  def _step(self):
    self.solver.integrate(self.tstop, step=True)
    self.nstep += 1
    self._t = self.solver.t
    self.jvec = self.solver.y[:3]
    self.evec = self.solver.y[3:]
    self.update()

  def integrator_setup(self):
    '''Set up the integrator.'''

    # Integration parameters
    self.nstep = 0

    self._y = list(np.r_[self.jvec, self.evec])

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.algo, nsteps=1, atol=self.atol, 
      rtol=self.rtol)
    self.solver.set_initial_value(self._y, self._t)
    if self.algo == 'vode':
      self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def evolve(self):
    '''Integrate the triple in time.'''
    self.printout()
    self.tstart = time.time()
    while ((self.t < self.tstop) and 
      (time.time() - self.tstart < self.cputstop)):

      self._step()
      if self.nstep % self.outfreq == 0:
        self.printout()

    self.printout()
    if self.outfilename is not None:
      self.outfile.close()

  def extrema(self):
    '''Integrate the triple, but only print out on eccentricity extrema.'''
    t_prev = 0
    e_prev = 0
    e_prev2 = 0
    while self.t < self.tstop:
      self._step()
      e = np.linalg.norm(self.evec)
      if e_prev2 < e_prev > e:
        outstring = ' '.join(map(str, [t_prev, e_prev]))
        if self.outfilename is None:
          print outstring
        else:
          self.outfile.write(outstring + '\n')
      t_prev = self.t
      e_prev2 = e_prev
      e_prev = e

    if self.outfilename is not None:
      self.outfile.close()

  def find_flips(self):
    '''Integrate the triple, but print out only when there is a flip.'''
    t_prev = 0
    e_prev = 0
    e_prev2 = 0
    sign_prev = np.sign(self.jvec[2])
    while self.t < self.tstop:
      self._step()
      e = np.linalg.norm(self.evec)
      if e_prev2 < e_prev > e:
        if np.sign(self.jvec[2]) != sign_prev:
          outstring = ' '.join(map(str, [t_prev, e_prev]))
          if self.outfilename is None:
            print outstring
          else:
            self.outfile.write(outstring + '\n')
        sign_prev = np.sign(self.jvec[2])
      t_prev = self.t
      e_prev2 = e_prev
      e_prev = e
    self.outfile.close()

  def printout(self):
    '''Print out the state of the system in the format:

    time  jx  jy  jz  ex  ey  ez

    '''

    outstring = ' '.join(map(str, np.concatenate((np.array([self.t]), 
      self.jvec, self.evec))))
    if self.outfilename is None:
      print outstring
    else:
      self.outfile.write(outstring + '\n')

  def flip_times(self, nflips=3):
    '''Find the times that the inner binary flips.'''
    sign = np.sign(self.jvec[2])
    sign_prev = sign
    flip_count = 0

    # Integrate along...
    while flip_count < nflips:
      self._step()
      sign = np.sign(self.jvec[2])
      if sign != sign_prev:
        flip_count += 1
        self.printout()
      sign_prev = sign

  def flip_period(self, nflips=3):
    '''Return the period of flips.'''
    sign = np.sign(self.jvec[2])
    sign_prev = sign
    flip_count = 0
    fliptime_prev = 0
    periods = []

    # Integrate along...
    while len(periods) < nflips:
      self._step()
      sign = np.sign(self.jvec[2])
      if sign != sign_prev:
        if fliptime_prev != 0:
          periods.append(self.t - fliptime_prev)
        fliptime_prev = self.t
      sign_prev = sign

    return np.mean(periods)

def _evec_root(x, j, g1):
  '''The set of equations that determine evec.'''

  # Orthogonal to j
  cond1 = x[0] * j[0] + x[1] * j[1] + x[2] * j[2]

  # Normalized
  cond2 = x[0]**2 + x[1]**2 + x[2]**2 - 1.

  # Gives the right argument of periapsis
  crossnorm = np.sqrt(j[0]**2 + j[1]**2)
  cond3 = x[0] * j[1] / crossnorm - x[1] * j[0] / crossnorm + cos(g1)

  return [cond1, cond2, cond3]
