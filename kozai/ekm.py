#! /usr/bin/env python

'''
ekm

Numerically integrate only the octupole term of the equations of motion of
a hierarchical triple.  This procedure averages over not only the
individual orbits, but also the individual KL cycles as well.
'''

# System packages
import json
import time

# Numerical packages
from math import pi, sin, cos, sqrt
import numpy as np
from scipy.integrate import ode, quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe

# Triplesec packages
from ts_constants import *

class TripleOctupole(object):
  '''A hierachical triple where only the octupole term of the Hamiltonian is
  considered.  The quadrupole term is averaged over.

  Parameters:
    e1: Inner eccentricity
    e2: Outer eccentricity
    a1: Inner semi-major axis in AU
    a2: OUter semi-major axis in AU
    inc: Inclination in degrees
    g1: Argument of periastron in degrees
    Omega: Longitude of ascending node in degrees

  Other parameters:
    epsoct: e2 / (1 - e2^2) * (a1 / a2)
    phiq: The value of the quadrupole term of the Hamiltonian
    chi: The other integral of motion of the octupole term

    tstop: The time to integrate (units of t_KL)
    cputstop: The number of CPU seconds to integrate for
    outfreq: Print out state every n steps (-1 for no output)
    outfile: Filename to write output to (None for stdout)
    atol: Absolute tolerance of the integrator
    rtol: Relative tolerance of the integrator
  '''

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, Omega=180,
    g1=0, epsoct=None, phiq=None, chi=None):

    #
    # Given parameters
    #
    self.a1 = float(a1)
    self.a2 = float(a2)
    self.e1 = e1
    self.e2 = e2
    self.inc = inc
    self.Omega = Omega
    self.g1 = g1

    #
    # Derived parameters
    #
    self.cosi = np.cos(inc * np.pi / 180)
    self.j = np.sqrt(1 - e1**2)
    self.jz = self.j * self.cosi

    # We don't use calc_CKL() at first because we need CKL to calculate chi.
    self.CKL = (self.e1**2 * (1 - 5/2. * (1 - self.cosi**2) *
      np.sin(self.omega)**2))

    #
    # Integration parameters
    #
    self.tstop = None
    self.t = 0
    self.nstep = 0
    self.cputstop = 300
    self.outfreq = 1
    self.integration_algo = 'vode'
    self.atol = 1e-9
    self.rtol = 1e-9

  @property
  def a1(self):
    return self._a1 / au

  @a1.setter
  def a1(self, val):
    if val is not None:
      self._a1 = val * au
    else:
      self._a1 = None

  @property
  def a2(self):
    return self._a2 / au

  @a2.setter
  def a2(self, val):
    if val is not None:
      self._a2 = val * au
    else:
      self._a2 = None

  @property
  def inc(self):
    return self._inc * 180 / pi

  @inc.setter
  def inc(self, val):
    self._inc = val * pi / 180

  @property
  def Omega(self):
    return self._Omega * 180 / pi

  @Omega.setter
  def Omega(self, val):
    self._Omega = val * pi / 180

  @property
  def g1(self):
    return self._g1 * 180 / pi

  @g1.setter
  def g1(self, val):
    self._g1 = val * pi / 180

  @property
  def phiq(self):
    return self.CKL + self.jz**2 / 2.

  @phiq.setter
  def phiq(self, val):
    self.CKL = val
    self.jz = 0

  @property
  def chi(self):
    return F(self.CKL) - self.epsoct * cos(self.Omega)

  @chi.setter
  def chi(self, val):
    self.Omega = np.arccos((F(self.CKL) - self.chi) / self.epsoct)

  @property
  def x(self):
    return (3 - 3 * self.CKL) / (3 + 2 * self.CKL)

  @property
  def fj(self):
    return (15 * pi / (128 * sqrt(10)) / ellipk(self.x) * (4 - 11 * 
      self.CKL) * sqrt(6 + 4 * self.CKL))

  @property
  def fOmega(self):
    return ((6 * ellipe(self.x) - 3 * ellipk(self.x)) / (4 *
      ellipk(self.x)))

  def integrator_setup(self):
    '''Set up the integrator.'''

    # Integration parameters
    self.nstep = 0

    self._y = [self.jz, self.Omega]

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.algo, nsteps=1, atol=self.atol, 
      rtol=self.rtol)
    self.solver.set_initial_value(self._y, self._t)
    if self.algo == 'vode':
      self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def _deriv(self, t, y):
    # Eqs. 11 of Katz (2011)
    jz, Omega = y

    jzdot = -self.epsoct * self.fj * sin(Omega)
    Omegadot = jz * self.fOmega

    return [jzdot, Omegadot]

  def _step(self):
    self.solver.integrate(self.tstop, step=True)
    self.t = self.solver.t
    self.jz, self.Omega = self.solver.y
    self.nstep += 1

  def set_CKL(self):
    self.CKL = self.calc_CKL()

  def calc_CKL(self):
    return self.phiq - self.jz**2 / 2.

  def evolve(self, tstop):
    '''Integrate the triple in time.'''
    self.printout()

    self.tstart = time.time()
    while ((self.t < self.tstop) and 
      ((time.time() - self.tstart) < self.cputstop)):
      self._step()
      if self.nstep % self.outfreq == 0:
        self.printout()

    self.printout()
    if self.outfilename is not None:
      self.outfile.close()

  def period(self):
    '''Analytically calculate the period of EKM oscillations.'''

    # First calculate the limits. 
    xcrit = brentq(lambda x: ellipk(x) - 2 * ellipe(x), 0, 1)
    phicrit = 3 * (1 - xcrit) / (3 + 2 * xcrit)

    if self.phiq < phicrit:
      CKLmin = brentq(lambda CKL: self.chi - self.epsoct - F(CKL), self.tol, self.phiq)
    else:
      # Check if flips occur for Omega = Pi or 0
      if (np.sign(self.chi - self.epsoct - F(self.tol)) != 
          np.sign(self.chi - self.epsoct - F(self.phiq))):
        CKLmin = brentq(lambda CKL: self.chi - self.epsoct - F(CKL), self.tol, self.phiq)
      else:
        CKLmin = brentq(lambda CKL: self.chi + self.epsoct - F(CKL), self.tol, self.phiq)
    if self.doesflip():
      CKLmax = self.phiq
    else:
      CKLmax = brentq(lambda CKL: self.chi + self.epsoct - F(CKL), 0, 1)

    prefactor = 256 * np.sqrt(10) / (15 * np.pi) / self.epsoct
    P = quad(lambda CKL: (prefactor * ellipk((3 - 3*CKL)/(3 + 2*CKL)) / 
      (4 - 11*CKL) / np.sqrt(6 + 4*CKL) / np.sqrt(1 - 1/self.epsoct**2 *
      (F(CKL) - self.chi)**2) / np.sqrt(2* np.fabs(self.phiq - CKL))), 
      CKLmin, CKLmax, epsabs=1e-12, epsrel=1e-12, limit=100)

    return P[0]

  def numeric_period(self, n_flips=3):
    '''Calculate the period of EKM oscillations by integrating the EOMs and
    taking the average flip time for n_flips flips.'''

    t_flip_prev = 0
    sign_prev = np.sign(self.jz)
    periods = []
    while (len(periods) < n_flips) and (self.t < self.tstop):
      self._step()
      if np.sign(self.jz) != sign_prev:
        if t_flip_prev != 0:
          periods.append(self.t - t_flip_prev)
        t_flip_prev = self.t
        sign_prev = np.sign(self.jz)

    return np.mean(periods)

  def doesflip(self):
    '''Return True if the triple flips, false otherwise.  This is determined
    using Eq. 16 of Katz (2011).
    '''

    #
    # Calculate Delta F over many values of x.  x can range from
    #
    # C_KL < x < C_KL + j_z^2 / 2
    #
    X = np.linspace(self.CKL, self.CKL + (self.jz)**2 / 2.)
    DeltaF = np.fabs(np.array(map(F, X)) - F(self.CKL))

    epsoct_crit = np.max(DeltaF) / 2.

    if self.epsoct > epsoct_crit:
      return True
    else:
      return False

  def state(self):
    '''Print out the state of the system in the format:

    time  jz  Omega  <f_j>  <f_Omega>  x  C_KL

    '''

    return (self.t, self.jz, self.Omega, self.fj, self.fOmega, self.x,
      self.CKL)

def _F_integrand(x):
  return (ellipk(x) - 2 * ellipe(x)) / (41*x - 21) / np.sqrt(2*x + 3)

def F(CKL):
  x_low = (3 - 3 * CKL) / (3 + 2 * CKL)
  integral = quad(_F_integrand, x_low, 1)[0]
  return 32 * sqrt(3) / pi * integral
