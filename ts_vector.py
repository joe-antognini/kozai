#! /usr/bin/env python

import numpy as np
from math import sin, cos
from scipy.integrate import ode, quad

class Triple_vector:
  '''Evolve a triple in time using the vectorial equations of motion.'''

  def __init__(self, a1=1, a2=20, e1=.1, e2=.3, inc=80, longascnode=180,
    argperi=0, m1=1, m3=1, epsoct=None, phiq=None, Xi=None, tstop=1e5,
    cputstop=300, outfreq=1, outfilename=None, atol=1e-9, rtol=1e-9):

    # Given parameters
    self.a1 = a1
    self.a2 = a2
    self.e1 = e1
    self.e2 = e2
    self.inc = inc * np.pi / 180
    self.Omega = longascnode * np.pi / 180
    self.omega = argperi * np.pi / 180
    self.m1 = m1
    self.m3 = m3

    # Derived parameters
    self.Omega_e = self.Omega + self.omega
    self.ie = np.arcsin(cos(self.omega) / (sin(self.inc) * (cos(self.Omega
      + self.omega) * cos(self.Omega) - sin(self.Omega + self.omega) *
      sin(self.Omega))))
    self.j = np.sqrt(1 - self.e1**2)

    if epsoct is None:
      self.epsoct = (self.a1 / self.a2) * self.e2 / (1 - self.e2**2)
    else:
      self.epsoct = epsoct

    # The vectorial elements
    self.jvec = self.j * np.array([
      sin(self.inc) * sin(self.Omega),
      -sin(self.inc) * cos(self.Omega),
      cos(self.inc)])
    self.evec = self.e1 * np.array([
      sin(self.ie) * cos(self.Omega_e),
      sin(self.ie) * sin(self.Omega_e),
      cos(self.ie)])

    # Elements of the potential
    self.Phi0 = 4 * np.pi**2 * self.m3 * self.a1**2 / (self.a2**3 * (1 -
      self.e2**2)**(3/2.))
    self.phiq = 3/4. * (self.jvec[2]**2 / 2. + self.e1**2 - 5/2. *
      self.evec[2]**2 - 1/6.)
    self.phioct = self.epsoct * 75/64. * (self.evec[0] * (1/5. - 8/5. * 
      self.e1**2 + 7 * self.evec[2]**2 - self.jvec[2]**2) - 2 * self.evec[2]
      * self.jvec[0] * self.jvec[2])

    # Integration parameters
    self.nstep = 0
    self.t = 0
    self.tstop = tstop
    self.cputstop = cputstop
    self.outfreq = outfreq
    self.outfilename = outfilename
    self.integration_algo = 'vode'
    self.y = np.concatenate((self.jvec, self.evec))

    if self.outfilename is not None:
      self.outfile = open(self.outfilename, 'w')

    # Set up the integrator
    self.solver = ode(self._deriv)
    self.solver.set_integrator(self.integration_algo, nsteps=1, atol=atol,
      rtol=rtol)
    self.solver.set_initial_value(self.y, self.t).set_f_params(self.epsoct,
      self.phiq)
    self.solver._integrator.iwork[2] = -1 # Don't print FORTRAN errors

  def _deriv(self, t, y, epsoct):
    '''The EOMs.  See Eqs. 4 of Katz et al. (2011).'''

    # Note that we have the following correspondences:
    # y[0]  y[1]  y[2]  y[3]  y[4]  y[5]
    # j_x   j_x   j_z   e_x   e_y   e_z

    #The total eccentricity:
    e_sq = y[3]**2 + y[4]**2 + y[5]**2

    # Calculate the derivatives of phi.
    grad_j_phi_q = np.array([0, 0, 3/4. * y[2]])
    grad_j_phi_oct = -75/32. * np.array([y[5] * y[2], 0,
      y[3] * y[2] + y[5] * y[0]])
    grad_e_phi_q = np.array([3/2. * y[3], 3/2. * y[4], -9/4. * y[5]])
    grad_e_phi_oct = np.array([
      75/64. * (1/5. - 8/5. * e_sq + 7 * y[5]**2 - y[2]**2) - 15/4. * y[3]**2,
      -15/4. * y[3] * y[4],
      75/64. * (54/5. * y[3] * y[5] - 2 * y[0] * y[2])])

    grad_j_phi = grad_j_phi_q + epsoct * grad_j_phi_oct
    grad_e_phi = grad_e_phi_q + epsoct * grad_e_phi_oct

    djdtau = np.cross(y[:3], grad_j_phi) + np.cross(y[3:], grad_e_phi)
    dedtau = np.cross(y[:3], grad_e_phi) + np.cross(y[3:], grad_j_phi)

    return np.concatenate(djdtau, dedtau)

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

  def printout(self):
    '''Print out the state of the system in the format:

    time  jx  jy  jz  ex  ey  ez

    '''

    outstring = ' '.join(map(str, np.concatenate((self.t, self.jvec,
      self.evec))))
    if self.outfilename is None:
      print outstring
    else:
      self.outfile.write(outstring + '\n')

  def __exit__(self):
    self.outfile.close()

