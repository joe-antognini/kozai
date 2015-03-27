#! /usr/bin/env python

import numpy as np
from math import sin, cos
from scipy.integrate import ode, quad
from scipy.optimize import root, fsolve

class Triple_vector:
  '''Evolve a triple in time using the vectorial equations of motion.'''

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, longascnode=180,
    argperi=0, m1=1, m3=1, epsoct=None, tstop=1e3, cputstop=300, outfreq=1,
    outfilename=None, atol=1e-9, rtol=1e-9):

    # Given parameters
    self.a1 = a1
    self.a2 = a2
    self.e1_0 = e1
    self.e2_0 = e2
    self.inc_0 = inc * np.pi / 180
    self.Omega_0 = longascnode * np.pi / 180
    self.omega_0 = argperi * np.pi / 180
    self.m1 = m1
    self.m3 = m3

    # Derived parameters
    self.j_0 = np.sqrt(1 - self.e1_0**2)

    if epsoct is None:
      self.epsoct = (self.a1 / self.a2) * self.e2 / (1 - self.e2**2)
    else:
      self.epsoct = epsoct
      self.e2_0 = None
      self.a1 = None
      self.a2 = None

    # The vectorial elements
    self.jhatvec_0 = np.array([
      sin(self.inc_0) * sin(self.Omega_0),
      -sin(self.inc_0) * cos(self.Omega_0),
      cos(self.inc_0)])
    self.jvec_0 = self.j_0 * self.jhatvec_0

    ehatvec_sol = root(_evec_root, [.5, .5, .5], (self.jhatvec_0, self.omega_0))
    self.ehatvec_0 = ehatvec_sol.x
    self.evec_0 = self.e1_0 * self.ehatvec_0

    # Elements of the potential
#    self.Phi0 = 4 * np.pi**2 * self.m3 * self.a1**2 / (self.a2**3 * (1 -
#      self.e2**2)**(3/2.))
    self.phiq = 3/4. * (self.jvec_0[2]**2 / 2. + self.e1_0**2 - 5/2. *
      self.evec_0[2]**2 - 1/6.)
    self.phioct = self.epsoct * 75/64. * (self.evec_0[0] * (1/5. - 8/5. * 
      self.e1_0**2 + 7 * self.evec_0[2]**2 - self.jvec_0[2]**2) - 2 *
      self.evec_0[2] * self.jvec_0[0] * self.jvec_0[2])

    # Integration parameters
    self.nstep = 0
    self.t = 0
    self.tstop = tstop
    self.cputstop = cputstop
    self.outfreq = outfreq
    self.outfilename = outfilename
    self.integration_algo = 'vode'
    self.y = list(np.concatenate((self.jvec_0, self.evec_0)))

    # We have saved some of the initial values (e.g., jvec_0).  Here we set
    # them to their respective parameters.  (I.e., we set jvec = jvec_0[:].)
    self._set_params()

    if self.outfilename is not None:
      self.outfile = open(self.outfilename, 'w')

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.integration_algo, nsteps=1, atol=atol,
      rtol=rtol)
    self.solver.set_initial_value(self.y, self.t).set_f_params(self.epsoct)
    self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def _set_params(self):
    '''Set the variables to their initial values.  Just a clone of
    reset().'''
    self.reset()

  def reset(self):
    '''Set the variables to their initial values.'''

    # Ordinary variables
    self.e1 = self.e1_0
    self.e2 = self.e2_0
    self.inc = self.inc_0
    self.Omega = self.Omega_0
    self.omega = self.omega_0
    self.j = self.j_0
    self.nstep = 0
    self.t = 0

    # Arrays need to be deep copied
    self.jhatvec = self.jhatvec_0[:]
    self.jvec = self.jvec_0[:]
    self.ehatvec = self.ehatvec_0[:]
    self.evec = self.evec_0[:]

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
    self.t = self.solver.t
    self.jvec = self.solver.y[:3]
    self.evec = self.solver.y[3:]

  def integrate(self):
    '''Integrate the triple in time.'''
    self.printout()
    while self.t < self.tstop:
      self._step()
      if self.nstep % self.outfreq == 0:
        self.printout()

    self.printout()
    self.outfile.close()

  def ecc_extrema(self):
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

    self.outfile.close()

  def printflips(self):
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

  def __exit__(self):
    self.outfile.close()

def _evec_root(x, j, omega):
  '''The set of equations that determine evec.'''

  # Orthogonal to j
  cond1 = x[0] * j[0] + x[1] * j[1] + x[2] * j[2]

  # Normalized
  cond2 = x[0]**2 + x[1]**2 + x[2]**2 - 1.

  # Gives the right argument of periapsis
  crossnorm = np.sqrt(j[0]**2 + j[1]**2)
  cond3 = x[0] * j[1] / crossnorm - x[1] * j[0] / crossnorm + cos(omega)

  return [cond1, cond2, cond3]
