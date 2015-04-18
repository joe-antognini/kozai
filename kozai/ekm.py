#! /usr/bin/env python

'''
ekm

Numerically integrate only the octupole term of the equations of motion of
a hierarchical triple.  This procedure averages over not only the
individual orbits, but also the individual KL cycles as well.
'''

# System packages
import argparse
import json
import random
import sys
import time

# Numerical packages
import numpy as np
from scipy.integrate import ode, quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe

# Triplesec packages
from ts_constants import *

class TripleOctupole:
  '''A hierachical triple where only the octupole term of the Hamiltonian is
  considered.  The quadrupole term is averaged over.

  Parameters:
    e1: Inner eccentricity
    e2: Outer eccentricity
    a1: Inner semi-major axis in AU
    a2: OUter semi-major axis in AU
    inc: Inclination in degrees
    argperi: Argument of periastron in degrees
    longascnode: Longitude of ascending node in degrees

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

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, longascnode=180,
    argperi=0, epsoct=None, phiq=None, chi=None, tstop=1e3, cputstop=300, 
    outfreq=1, outfilename=None, atol=1e-9, rtol=1e-9,
    integration_algo='vode'):

    #
    # Given parameters
    #
    self.a1 = float(a1)
    self.a2 = float(a2)
    self.e1 = e1
    self.e2 = e2
    self.inc = inc
    self.Omega = longascnode * np.pi / 180
    self.omega = argperi * np.pi / 180

    #
    # Derived parameters
    #
    self.cosi = np.cos(inc * np.pi / 180)
    self.j = np.sqrt(1 - e1**2)
    self.jz = self.j * self.cosi

    # We don't use calc_CKL() at first because we need CKL to calculate chi.
    self.CKL = (self.e1**2 * (1 - 5/2. * (1 - self.cosi**2) *
      np.sin(self.omega)**2))

    if phiq is None:
      self.phiq = self.CKL + self.jz**2 / 2.
    else:
      self.phiq = phiq
      self.jz = 0
      self.CKL = self.phiq
    if epsoct is None:
      self.epsoct = self.e2 / (1 - self.e2**2) * (self.a1 / self.a2)
    else:
      self.epsoct = epsoct
    if chi is None:
      self.chi = F(self.CKL) - self.epsoct * np.cos(self.Omega)
    else:
      self.chi = chi
      self.Omega = np.arccos((F(self.CKL) - self.chi) / self.epsoct)

    self.set_x()
    self.set_fj()
    self.set_fOmega()

    #
    # Integration parameters
    #
    self.nstep = 0
    self.t = 0
    self.tstop = tstop
    self.cputstop = cputstop
    self.outfreq = outfreq
    self.outfilename = outfilename
    self.integration_algo = integration_algo
    self.y = [self.jz, self.Omega]
    self.tol = 1e-9
    self.atol = atol
    self.rtol = rtol

    if self.outfilename is not None:
      self.outfile = open(self.outfilename, 'w')

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.integration_algo, nsteps=1, 
      atol=self.atol, rtol=self.rtol)
    self.solver.set_initial_value(self.y, self.t).set_f_params(self.epsoct,
      self.phiq)
    self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def _deriv(self, t, y, epsoct, phiq):
    # Eqs. 11 of Katz (2011)
    jz, Omega = y
    CKL = phiq - jz**2 / 2.
    x = (3 - 3 * CKL) / (3 + 2 * CKL)
    fj = (15 * np.pi / (128 * np.sqrt(10)) / ellipk(x) * (4 - 11 * CKL) 
      * np.sqrt(6 + 4 * CKL))
    fOmega = ((6 * ellipe(x) - 3 * ellipk(x)) / (4 * ellipk(x)))

    jzdot = -epsoct * fj * np.sin(Omega)
    Omegadot = jz * fOmega

    return [jzdot, Omegadot]

  def _step(self):
    self.solver.integrate(self.tstop, step=True)

    self.t = self.solver.t
    self.jz, self.Omega = self.solver.y

    # Update all the parameters
    self.set_CKL()
    self.set_x()
    self.set_fj()
    self.set_fOmega()
    self.nstep += 1

  def set_CKL(self):
    self.CKL = self.calc_CKL()

  def calc_CKL(self):
    return self.phiq - self.jz**2 / 2.

  def calc_x(self):
    return (3 - 3 * self.CKL) / (3 + 2 * self.CKL)

  def set_x(self):
    self.x = self.calc_x()

  def calc_fj(self):
    return (15 * np.pi / (128 * np.sqrt(10)) / ellipk(self.x) *
      (4 - 11 * self.CKL) * np.sqrt(6 + 4 * self.CKL))

  def set_fj(self):
    self.fj = self.calc_fj()

  def calc_fOmega(self):
    return ((6 * ellipe(self.x) - 3 * ellipk(self.x)) / (4 *
      ellipk(self.x)))

  def set_fOmega(self):
    self.fOmega = self.calc_fOmega()

  def integrate(self):
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

  def printout(self):
    '''Print out the state of the system in the format:

    time  jz  Omega  <f_j>  <f_Omega>  x  C_KL

    '''
    outstring = ' '.join(map(str, [self.t, self.jz, self.Omega, self.fj,
      self.fOmega, self.x, self.CKL]))
    if self.outfilename is None:
      print outstring
    else:
      self.outfile.write(outstring + '\n')

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

  def __exit__(self):
    self.outfile.close()

def _F_integrand(x):
  return (ellipk(x) - 2 * ellipe(x)) / (41*x - 21) / np.sqrt(2*x + 3)

def F(CKL):
  x_low = (3 - 3 * CKL) / (3 + 2 * CKL)
  integral = quad(_F_integrand, x_low, 1)[0]
  return 32 * np.sqrt(3) / np.pi * integral
