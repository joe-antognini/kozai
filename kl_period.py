#! /usr/bin/env python

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from math import sqrt, cos
import numpy as np
from scipy.integrate import quad
from triplesec import triplesec_step
import astropy.constants as const
import astropy.units as u
import json

G = const.G.value
c = const.c.value
yr2s = 3.15576e7
au = const.au.value

def kl_period_oom(a1, a2, e2, m1, m3, m2=0):
  '''The usual KL period formula:

  t_KL = P_out^2 / P_in * (1 - e2^2)^(3/2)

  Parameters:
    a1: Inner semi-major axis in AU
    a2: Outer semi-major axis in AU
    e2: Outer eccentricity
    m1: Mass of the primary of the inner binary in M_Sun
    m3: Mass of the tertiary in M_Sun
    m2: Mass of the secondary of the inner binary in M_Sun

  Returns:
    P: The period of KL oscillations in years.
  '''

  P_out2 = a2**3 / (m1 + m2 + m3)
  P_in = sqrt(a1**3 / (m1 + m2))

  return P_out2 / P_in * (1 - e2**2)**(3./2)

def is_librating(e1, inc, g1):
  '''Determine whether the triple is librating or rotating.

  Parameters:
    e1:  Inner eccentricity
    inc: Inclination in degrees
    g1:  Inner argument of periapsis in degrees

  Returns:
    True if librating, False if rotating.
  '''

  eps2 = 1 - e1**2
  Th = eps2 * cos(inc * np.pi / 180)**2
  Hhat = ((5 - 3 * eps2) * (eps2 - 3 * Th) - 15 * (1 - eps2) * (eps2 - Th) *
    cos(2*g1*np.pi/180)) / eps2

  print Hhat
  print Th

  if Hhat + 6 * Th - 2 > 0:
    return True
  else:
    return False

def depsdh(eps, H, Th):
  '''The derivative of epsilon with respect to H.'''

#  print ((3*eps**4 + eps**2 * (H - 9 * Th - 5) + 15*Th)**2 / (225 *
#    (1-eps**2)**2 * (eps**2 - Th)**2))

  return (eps**2 / (30 * (1-eps**2) * (eps**2 - Th) * sqrt(1 - (3*eps**4 +
    eps**2 * (H - 9 * Th - 5) + 15*Th)**2 / (225 * (1-eps**2)**2 * (eps**2 -
    Th)**2))))

def kl_period_norm(Hhat, Th):
  zeta = 20 - Hhat + 24 * Th
  epsmin = 1/6. * sqrt(zeta - sqrt(zeta**2 - 2160 * Th))

  # Check whether the triple is librating or rotating
  if Hhat + 6 * Th - 2 > 0:
    epsmax = 1/6. * sqrt(zeta + sqrt(zeta**2 - 2160 * Th))
  else:
    epsmax = sqrt((Hhat + 6 * Th + 10) / 12.)

  return quad(depsdh, epsmin, epsmax, args=(Hhat, Th), epsabs=1e-13, epsrel=1e-13)[0]

def kl_tp_period(a1, a2, e1, e2, inc, m1, m3, g):
  '''Return the Kozai period in years.

  Parameters:
    a1:  Inner semi-major axis in AU
    a2:  Outer semi-major axis in AU
    e1:  Inner eccentricity
    e2:  Outer eccentricity
    inc: Mutual inclination in degrees
    m1:  Mass of the inner binary in M_Sun
    m3:  Mass of the tertiary in M_Sun
    g:   Argument of periapsis in degrees

  Returns:
    P: The period in yr
  '''

  L1toC2 = (16 * a2 * (1 - e2**2)**(3/2.) / m3 * (a2 / a1)**2 * sqrt(m1 * a1)
    / (2 * np.pi))

  th = cos(np.pi * inc / 180)
  Th = th**2 * (1 - e1**2)
  Hhat = ((2 + 3*e1**2) * (1 - 3*th**2) - 15 * e1**2 * (1 - th**2) * 
    cos(2 * g * np.pi / 180))

  print Hhat
  print Th

  return 2 * L1toC2 * kl_period_norm(Hhat, Th)

def numerical_kl_period(m, e, a, g, inc, nperiods=10, cputstop=300, 
  tstop_factor=100, in_params=(1, (1e-11, 1e-11), (False, False, False))):
  '''Calculate the period of KL oscillations by explicitly integrating the
  secular equations of motion.

  Input:
    m: A list containing the three masses in solar masses (m0, m1, m2)
    e: A list containing the eccentricities (e1, e2)
    a: A list containing the semi-major axes in AU (a1, a2)
    g: A list containing the arguments of periapsis in degrees (g1, g2)
    inc: The mutual inclination in degrees
    nperiods: The number of KL periods to integrate for
    cputstop: The maximum wall time to allow for integration in sec
    tstop_factor: The maximum factor greater than the standard KL timescale
                  to allow for.
    in_params: A tuple containing:
      outfreq: How many steps between printing out the state
               (-1 for no printing)
      acc: A tuple containing the accuracy targets (relacc, absacc)
      terms: A tuple saying what terms to include (gc, oct, hex)

  Output:
    The average period of KL oscillations in yr.
  '''

  e_prev2 = 0.
  e_prev  = 0.

  periods = []
  emin_tstart = 0
  emax_tstart = 0

  # Prepare the inputs for triplesec_step (convert units, etc.)
  p_in  = a[0]**(3./2) / sqrt(m[0] + m[1])
  p_out = a[1]**(3./2) / sqrt(m[0] + m[1] + m[2])
  tstop = (tstop_factor * p_out**2 / p_in * (1 - e[1]**2)**(3/2) * yr2s, cputstop)
  m = [elem * const.M_sun.value for elem in m]
  a = [elem * const.au.value for elem in a]
  g = [elem * np.pi / 180 for elem in g]
  inc *= np.pi / 180
  r = [0, 0]

  for step in triplesec_step(m, r, e, a, g, inc, tstop, in_params):
    if len(periods) == 2 * nperiods:
      break
    e = step[0][3]
    t = step[0][0]
    if e_prev2 < e_prev > e:
      if emax_tstart > 0:
        periods.append(t - emax_tstart)
      emax_tstart = t
    elif e_prev2 > e_prev < e:
      if emin_tstart > 0:
        periods.append(t - emin_tstart)
      emin_tstart = t

    prevstep = step[0]
    e_prev2 = e_prev
    e_prev = e

  return np.mean(periods) * (u.s / u.yr).to(1)

if __name__ == '__main__':
  print kl_period_oom(1, 100, .9, 1, 1)
  print kl_tp_period(1, 100, .05, .9, 85, 1, 1, 0)
  print numerical_kl_period([1, 1e-6, 1], [.05, .9], [1, 100], [0, 0], 85)
