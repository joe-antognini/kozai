"""Code to model the eccentric Kozai mechanism.

Numerically integrate only the octupole term of the equations of motion of a
hierarchical triple.  This procedure averages over not only the individual
orbits, but also the individual KL cycles as well.

"""

import json
import time
from math import acos, cos, pi, sin, sqrt

import numpy as np
from scipy.integrate import ode, quad
from scipy.optimize import brentq
from scipy.special import ellipk, ellipe


class TripleOctupole:
    """A hierarchical triple with only the octupole term in the Hamiltonian.

    The quadrupole term is averaged over in this triple.

    Args:
        e1: Inner eccentricity
        inc: Inclination in degrees
        g1: Argument of periastron in degrees
        Omega: Longitude of ascending node in degrees
        epsoct: The relative strength of the octupole to quadrupole terms

    Attributes:
        phiq: The value of the quadrupole term of the Hamiltonian
        chi: The other integral of motion of the octupole term
        tstop: The time to integrate (units of t_KL)
        cputstop: The number of CPU seconds to integrate for
        outfreq: Print out state every n steps (-1 for no output)
        outfile: Filename to write output to (None for stdout)
        atol: Absolute tolerance of the integrator
        rtol: Relative tolerance of the integrator

  """

    def __init__(
        self, e1=0.1, inc=80, Omega=180, g1=0, epsoct=1e-2, phiq=None, chi=None
    ):

        if (phiq is None and chi is not None) or (
            phiq is not None and chi is None
        ):
            raise ValueError('Either phiq and chi must both be set or neither')

        self.epsoct = epsoct

        if phiq is None:
            _inc = inc * pi / 180
            self.e1 = e1
            self.Omega = Omega
            self.g1 = g1

            CKL = self.e1**2 * (
                1 - 5 / 2.0 * sin(_inc)**2 * sin(self._g1)**2
            )
            self.jz = sqrt(1 - self.e1**2) * cos(_inc)

            self._phiq = CKL + self.jz**2 / 2
            self._chi = F(CKL) - self.epsoct * cos(self._Omega)

        else:
            self.phiq = phiq
            self.chi = chi
            self.e1 = None
            self._g1 = None

        # Integration parameters
        self.tstop = None
        self.t = 0
        self.nstep = 0
        self.cputstop = 300
        self.outfreq = 1
        self.algo = 'vode'
        self.atol = 1e-9
        self.rtol = 1e-9
        self.tol = 1e-5
        self.maxoutput = int(1e6)

        # Store the initial state
        self.save_as_initial()

    @property
    def chi(self):
        return self._chi

    @chi.setter
    def chi(self, val):
        self._chi = val
        self._Omega = acos((F(self.CKL) - val) / self.epsoct)

    @property
    def CKL(self):
        """The libration constant"""
        return self.phiq - self.jz**2 / 2.0

    @property
    def fj(self):
        return (
            15
            * pi
            / (128 * sqrt(10))
            / ellipk(self.x)
            * (4 - 11 * self.CKL)
            * sqrt(6 + 4 * self.CKL)
        )

    @property
    def fOmega(self):
        return (6 * ellipe(self.x) - 3 * ellipk(self.x)) / (4 * ellipk(self.x))

    @property
    def g1(self):
        return self._g1 * 180 / pi

    @g1.setter
    def g1(self, val):
        self._g1 = val * pi / 180

    @property
    def inc(self):
        return acos(self.jz / sqrt(1 - self.e1**2)) * 180 / pi

    @property
    def Omega(self):
        return self._Omega * 180 / pi

    @Omega.setter
    def Omega(self, val):
        self._Omega = val * pi / 180

    @property
    def phiq(self):
        return self._phiq

    @phiq.setter
    def phiq(self, val):
        self._phiq = val
        self.jz = 0

        self._a1 = None
        self._a2 = None
        self._e2 = None

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, val):
        self._t = val

    @property
    def x(self):
        return (3 - 3 * self.CKL) / (3 + 2 * self.CKL)

    def save_as_initial(self):
        """Set the current parameters as the initial parameters."""

        self.initial_state = {}
        self.initial_state['Omega'] = self.Omega
        self.initial_state['CKL'] = self.CKL
        self.initial_state['jz'] = self.jz
        self.initial_state['phiq'] = self.phiq
        self.initial_state['chi'] = self.chi

        if self.e1 is not None:
            self.initial_state['g1'] = self.g1
            self.initial_state['e1'] = self.e1
            self.initial_state['inc'] = self.inc

    def integrator_setup(self):
        """Set up the integrator."""

        # Integration parameters
        self.nstep = 0

        self._y = [self.jz, self._Omega]

        # Set up the integrator.
        self.solver = ode(self._deriv)
        self.solver.set_integrator(
            self.algo, nsteps=1, atol=self.atol, rtol=self.rtol
        )
        self.solver.set_initial_value(self._y, self._t)
        if self.algo == 'vode':
            self.solver._integrator.iwork[
                2
            ] = -1  # Don't print FORTRAN errors.

    def reset(self):
        """Reset the triple to its initial configuration.  This resets the
    orbital parameters and time, but does not reset the integration
    options."""
        self.t = 0
        if self.e1 is None:
            self.phiq = self.initial_state['phiq']
            self.chi = self.initial_state['chi']
        else:
            self.e1 = self.initial_state['e1']
            self.Omega = self.initial_state['Omega']
            self.g1 = self.initial_state['g1']
            _inc = self.initial_state['inc'] * pi / 180

            CKL = self.e1**2 * (
                1 - 5 / 2.0 * sin(_inc)**2 * sin(self._g1)**2
            )
            self.jz = sqrt(1 - self.e1**2) * cos(_inc)

            self._phiq = CKL + self.jz**2 / 2
            self._chi = F(CKL) - self.epsoct * cos(self._Omega)

    def _deriv(self, t, y):
        # Eqs. 11 of Katz (2011)
        jz, Omega = y

        jzdot = -self.epsoct * self.fj * sin(Omega)
        Omegadot = jz * self.fOmega

        return [jzdot, Omegadot]

    def _step(self):
        self.solver.integrate(self.tstop, step=True)
        self.t = self.solver.t
        self.jz, self._Omega = self.solver.y
        self.nstep += 1

    def evolve(self, tstop):
        """Integrate the triple in time.

    Parameters:
      tstop: The time to integrate in units of the secular time."""

        self.tstop = tstop
        n_columns = len(self.state())
        self.integrator_setup()
        self.integration_steps = np.zeros((self.maxoutput, n_columns))
        self.integration_steps[0] = self.state()

        self.tstart = time.time()
        while (self.t < tstop) and (
            (time.time() - self.tstart) < self.cputstop
        ):

            self._step()
            if self.nstep % self.outfreq == 0:
                self.integration_steps[
                    self.nstep // self.outfreq
                ] = self.state()

        laststep = (self.nstep // self.outfreq) + 1
        self.integration_steps[laststep] = self.state()

        return self.integration_steps[: laststep + 1]

    def flip_period(self):
        """Analytically calculate the period of EKM oscillations."""

        # First calculate the limits.
        xcrit = brentq(lambda x: ellipk(x) - 2 * ellipe(x), 0, 1)
        phicrit = 3 * (1 - xcrit) / (3 + 2 * xcrit)

        if self.phiq < phicrit:
            CKLmin = brentq(
                lambda CKL: self.chi - self.epsoct - F(CKL),
                self.tol,
                self.phiq,
            )
        else:
            # Check if flips occur for Omega = Pi or 0.
            if np.sign(self.chi - self.epsoct - F(self.tol)) != np.sign(
                self.chi - self.epsoct - F(self.phiq)
            ):
                CKLmin = brentq(
                    lambda CKL: self.chi - self.epsoct - F(CKL),
                    self.tol,
                    self.phiq,
                )
            else:
                CKLmin = brentq(
                    lambda CKL: self.chi + self.epsoct - F(CKL),
                    self.tol,
                    self.phiq,
                )
        if self.doesflip():
            CKLmax = self.phiq
        else:
            CKLmax = brentq(lambda CKL: self.chi + self.epsoct - F(CKL), 0, 1)

        prefactor = 256 * np.sqrt(10) / (15 * np.pi) / self.epsoct
        P = quad(
            lambda CKL: (
                prefactor
                * ellipk((3 - 3 * CKL) / (3 + 2 * CKL))
                / (4 - 11 * CKL)
                / np.sqrt(6 + 4 * CKL)
                / np.sqrt(1 - 1 / self.epsoct**2 * (F(CKL) - self.chi)**2)
                / np.sqrt(2 * np.fabs(self.phiq - CKL))
            ),
            CKLmin,
            CKLmax,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=100,
        )

        return P[0]

    def numeric_flip_period(self, n_flips=3, tstop=1e6):
        """Calculate the period of EKM oscillations.

        This will integrate the EOMs and taking the average flip time for
        `n_flips` flips.

        """
        self.tstop = tstop
        n_columns = len(self.state())
        self.integrator_setup()
        self.integration_steps = np.zeros((self.maxoutput, n_columns))
        self.integration_steps[0] = self.state()

        t_flip_prev = 0
        sign_prev = np.sign(self.jz)
        periods = []
        self.tstart = time.time()
        while (
            (len(periods) < n_flips)
            and (self.t < self.tstop)
            and ((time.time() - self.tstart) < self.cputstop)
        ):

            self._step()
            if np.sign(self.jz) != sign_prev:
                if t_flip_prev != 0:
                    periods.append(self.t - t_flip_prev)
                t_flip_prev = self.t
                sign_prev = np.sign(self.jz)

        return np.mean(periods)

    def doesflip(self):
        """Check whether the triple flips.

        A flip is defined to occur when the inclination passes through +/- 90
        degrees.  This is determined using Eq. 16 of Katz (2011).

        """
        # Calculate Delta F over many values of x.  x can range from
        #
        # $$
        # C_KL < x < C_KL + j_z^2 / 2
        # $$
        X = np.linspace(self.CKL, self.CKL + (self.jz)**2 / 2)
        DeltaF = np.fabs(np.vectorize(F)(X) - F(self.CKL))

        epsoct_crit = np.max(DeltaF) / 2.0

        return self.epsoct > epsoct_crit

    def state(self):
        """Print out the state of the system in the format:

        ```
        time  jz  Omega  <f_j>  <f_Omega>  x  C_KL
        ```

        """
        return (
            self.t,
            self.jz,
            self.Omega,
            self.fj,
            self.fOmega,
            self.x,
            self.CKL,
        )

    def __repr__(self):
        """Print out the initial values in JSON format."""

        # Get the initial state
        json_data = self.initial_state

        # Add some other properties
        json_data['tstop'] = self.tstop
        json_data['cputstop'] = self.cputstop
        json_data['outfreq'] = self.outfreq
        json_data['atol'] = self.atol
        json_data['rtol'] = self.rtol
        json_data['algo'] = self.algo
        json_data['maxoutput'] = self.maxoutput

        return json.dumps(json_data, sort_keys=True, indent=2)


def _F_integrand(x):
    return (ellipk(x) - 2 * ellipe(x)) / (41 * x - 21) / np.sqrt(2 * x + 3)


def F(CKL):
    x_low = (3 - 3 * CKL) / (3 + 2 * CKL)
    integral = quad(_F_integrand, x_low, 1)[0]

    return 32 * sqrt(3) / pi * integral
