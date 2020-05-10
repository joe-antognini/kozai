import unittest

import numpy as np
from numpy.testing import assert_allclose
from kozai.delaunay import TripleDelaunay
from kozai.vectorial import TripleVectorial
from kozai.kl_period import is_librating
from kozai.kl_period import kl_period
from kozai.kl_period import kl_period_oom
from kozai.kl_period import numeric_kl_period
from kozai.kl_period import P_in
from kozai.kl_period import P_out


class TestKlPeriod(unittest.TestCase):
    def test_P_out(self):
        """Test the P_out calculation."""
        tv = TripleVectorial(a2=20, m1=1, m3=1)
        assert_allclose(P_out(tv), 63.2455532)

    def test_P_in(self):
        """Test the P_in calculation."""
        tv = TripleVectorial(a1=1, m1=1)
        assert_allclose(P_in(tv), 1)

    def test_klperiod_oom(self):
        """Test the OOM period of KL oscillations."""
        tv = TripleVectorial(a1=1, a2=20, m1=1, m3=1, e2=0.3)
        assert_allclose(kl_period_oom(tv), 1178.965050)

    def test_islibrating(self):
        """Test the islibrating calculation."""
        tv = TripleVectorial(e1=0.1, inc=80, g1=45)
        self.assertTrue(is_librating(tv))
        tv = TripleVectorial(e1=0.1, inc=80, g1=0)
        self.assertFalse(is_librating(tv))

    def test_kl_period_vectorial(self):
        """Test the KL period semi-analytic calculation."""
        tv = TripleVectorial(
            a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m3=1, g1=0, Omega=np.pi
        )
        assert_allclose(
            kl_period(tv), 4195.8240184679735, atol=1e-5, rtol=1e-5
        )

    def test_kl_period_delaunay(self):
        """Test the KL period semi-analytic calculation."""
        t = TripleDelaunay(
            a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m2=1, m3=1, g1=0, g2=0, inc=70
        )
        assert_allclose(kl_period(t), 4393.54571456, atol=1e-5, rtol=1e-5)

    def test_numerical_kl_period_vectorial(self):
        """Test against explicit integration of the equations of motion.

        See if the numerical calculation of the KL period by explicitly
        integrating the equations of motion works.

        """
        # Triple_vector test
        tv = TripleVectorial(
            a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m3=1, g1=0, Omega=180
        )
        tv.octupole = False
        assert_allclose(
            numeric_kl_period(tv, nperiods=3),
            4201.6634127,
            rtol=1e-3,
            atol=1e-3,
        )

    def test_numerical_kl_period_delaunay(self):
        """Test against explicit integration of the equations of motion.

        See if the numerical calculation of the KL period by explicitly
        integrating the equations of motion works.

        """
        # Triple test
        t = TripleDelaunay(
            a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m2=1, m3=1, g1=0, g2=0
        )
        t.octupole = False
        assert_allclose(
            numeric_kl_period(t, nperiods=3), 5866.384706, rtol=1e-3, atol=1e-3
        )
