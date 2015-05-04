#! /usr/bin/env python

'''
ts_vector

Numerically integrate the dynamics of a hierarchical triple in the test
particle limit using vectorial notation.
'''

# System packages
import json
import time

# Numerical packages
from math import acos, asin, cos, pi, sin, sqrt
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
    Omega=180., r1=0, r2=0):

    # First set the vectorial elements
    inc *= pi / 180
    Omega *= pi / 180
    self.jvec = sqrt(1 - e1**2) * np.array([
      sin(inc) * sin(Omega),
      -sin(inc) * cos(Omega),
      cos(inc)])

    ehatvec_sol = root(_evec_root, [.5, .5, .5], (self.jhatvec, g1 * 
      pi / 180))
    self.evec = e1 * ehatvec_sol.x

    self.a1 = a1
    self.a2 = a2
    self.e2 = e2
    self.m1 = m1
    self.m2 = 0
    self.m3 = m3
    self.r1 = r1
    self.r2 = r2
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
  def _g1(self):
    '''The argument of periapsis in radians'''
    crossvec = np.cross(np.array([0, 0, 1]), self.jhatvec)
    return acos(np.dot(self.ehatvec, crossvec / np.linalg.norm(crossvec)))

  @property
  def g1(self):
    '''g1 in degrees'''
    return self._g1 * 180 / pi

  @property
  def _Omega(self):
    '''The longitude of ascending node in radians'''
    try:
      return acos(-self.jhatvec[1] / sin(self._inc))
    except ValueError:
      return asin(self.jhatvec[0] / sin(self._inc))

  @property
  def Omega(self):
    '''The longitude of ascending node in degrees'''
    return self._Omega * 180 / pi

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
  def jhatvec(self):
    '''The normalized angular momentum vector'''
    return self.jvec / self.j

  @property
  def ehatvec(self):
    '''The normalized eccentricity vector'''
    return self.evec / self.e1

  @property
  def e1(self):
    '''The eccentricity of the inner binary'''
    return np.linalg.norm(self.evec)
  
  @property
  def j(self):
    '''The normalized angular momentum of the inner binary'''
    return np.linalg.norm(self.jvec)

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
      self.evec[2]**2 - 1/6.))

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

  def save_as_initial(self):
    '''Set the current parameters as the initial parameters.'''

    self.initial_state = {}
    self.initial_state['a1'] = self.a1
    self.initial_state['a2'] = self.a2
    self.initial_state['e1'] = self.e1
    self.initial_state['e2'] = self.e2
    self.initial_state['g1'] = self.g1
    self.initial_state['Omega'] = self.Omega
    self.initial_state['m1'] = self.m1
    self.initial_state['m3'] = self.m3
    self.initial_state['r1'] = self.r1
    self.initial_state['r2'] = self.r2
    self.initial_state['inc'] = self.inc
    self.initial_state['jhatvec'] = self.jhatvec[:]
    self.initial_state['ehatvec'] = self.ehatvec[:]
    self.initial_state['jvec'] = self.jvec[:]
    self.initial_state['evec'] = self.evec[:]

  def reset(self):
    '''Reset the triple to its initial configuration.  This resets the
    orbital parameters and time, but does not reset the integration
    options.'''
    self.t = 0

    # Arrays need to be deep copied
    self.jvec = self.initial_state['jvec'][:]
    self.evec = self.initial_state['evec'][:]

  def _deriv(self, t, y):
    '''The EOMs.  See Eqs. 4 of Katz et al. (2011).'''

    # Note that we have the following correspondences:
    # y[0]  y[1]  y[2]  y[3]  y[4]  y[5]
    # j_x   j_y   j_z   e_x   e_y   e_z

    # Unpack the values
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

    grad_j_phi = 0
    grad_e_phi = 0
    if self.quadrupole:
      grad_j_phi += grad_j_phi_q
      grad_e_phi += grad_e_phi_q
    if self.octupole:
      grad_j_phi += self.epsoct * grad_j_phi_oct
      grad_e_phi += self.epsoct * grad_e_phi_oct

    djdtau = np.cross(y[:3], grad_j_phi) + np.cross(y[3:], grad_e_phi)
    dedtau = np.cross(y[:3], grad_e_phi) + np.cross(y[3:], grad_j_phi)

    ret = np.r_[djdtau, dedtau]
    return list(ret)

  def _step(self):
    self.solver.integrate(self.tstop, step=True)
    self.nstep += 1
    self._t = self.solver.t
    self.jvec = self.solver.y[:3]
    self.evec = self.solver.y[3:]

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

  def evolve(self, tstop):
    '''Integrate the triple in time.'''

    self.tstop = tstop
    n_columns = len(self.state())
    self.integrator_setup()
    self.integration_steps = np.zeros((self.maxoutput, n_columns))
    self.integration_steps[0] = self.state()

    self.tstart = time.time()
    while ((self.t < self.tstop) and 
      (time.time() - self.tstart < self.cputstop)):

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
    '''Integrate the triple, but only save eccentricity extrema.'''

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

  def flip_period(self, nflips=3, tstop=1e10):
    '''Return the period of flips.'''

    self.tstop = tstop
    n_columns = len(self.state())
    self.integrator_setup()
    self.integration_steps = np.zeros((self.maxoutput, n_columns))

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

  def state(self):
    '''Return a tuple with the dynamical state of the system.

    Returns:
      (t, a1, e1, g1, a2, e2, Omega, inc)
    '''
    return (self.t, self.a1, self.e1, self.g1, self.a2, self.e2, 
      self.Omega, self.inc)
  
  def __repr__(self):
    '''Print out the initial values in JSON format.'''

    # Get the initial state
    json_data = self.initial_state
    for key in json_data:
      if type(json_data[key]) == np.ndarray:
        json_data[key] = list(json_data[key])

    # Add some other properties
    json_data['epsoct'] = self.epsoct
    json_data['tstop'] = self.tstop
    json_data['cputstop'] = self.cputstop
    json_data['outfreq'] = self.outfreq
    json_data['atol'] = self.atol
    json_data['rtol'] = self.rtol
    json_data['quadrupole'] = self.quadrupole
    json_data['octupole'] = self.octupole
    json_data['algo'] = self.algo
    json_data['maxoutput'] = self.maxoutput
    json_data['collision'] = self.collision

    return json.dumps(json_data, sort_keys=True, indent=2)

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
