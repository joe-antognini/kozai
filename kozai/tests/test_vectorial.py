import json
import unittest

from numpy.testing import assert_allclose

from kozai.vectorial import TripleVectorial


class TestObjectCreation(unittest.TestCase):
    def test_make_Triple_vector(self):
        """Try to create a Triple_vector class."""
        tv = TripleVectorial()
        self.assertIsNotNone(tv)

    def test_set_options(self):
        """Create a Triple_vector class with a few options."""
        tv = TripleVectorial(a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m3=3)
        self.assertIsNotNone(tv)

    def test_set_octupole(self):
        tv = TripleVectorial()
        tv.octupole = False
        self.assertIsNotNone(tv)

    def test_inc70(self):
        """Make a triple with inclination of 70 degrees."""
        tv = TripleVectorial(e1=0.05, inc=70)
        self.assertIsNotNone(tv)


class TestParameterCalculation(unittest.TestCase):
    def test_Th(self):
        """Test the calculation of Kozai's integral."""
        tv = TripleVectorial(e1=0.05, inc=70)
        assert_allclose(tv.Th, 0.11668533399441)

    def test_epsoct(self):
        """Test the epsoct calculation."""
        tv = TripleVectorial(a1=1, a2=20, e2=0.3)
        assert_allclose(tv.epsoct, 0.01648351648)

    def test_Phi0(self):
        """Test calculation of Phi0.  The calculation is done in units of Solar
    masses, years, and AU."""
        tv = TripleVectorial(a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m3=3)
        assert_allclose(tv.Phi0, 383313.6993558)

    def test_tsec(self):
        """Make sure the calculation of the secular timescale is correct."""
        tv = TripleVectorial(a1=1, a2=20, e1=0.1, e2=0.3, m1=1, m3=3)
        assert_allclose(tv.tsec, 11625553844.775972)

    def test_CKL(self):
        """Test the CKL calculation."""
        tv = TripleVectorial(e1=0.1, inc=80, g1=45)
        assert_allclose(tv.CKL, -0.00212307888)

    def test_Hhatquad(self):
        tv = TripleVectorial(e1=0.1, inc=80, g1=45)
        assert_allclose(tv.Hhatquad, 1.846364030293)


class TestRepresentation(unittest.TestCase):
    def test_repr(self):
        """Make sure that the triple can print its state."""
        t = TripleVectorial(inc=80, m1=1e-3)
        t.octupole = False
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['m1'], 1e-3)
        assert_allclose(state['inc'], 80)
        self.assertFalse(state['octupole'])

    def test_save_as_initial(self):
        """Try to set a parameter after creating the object and saving it as an
    initial condition."""
        t = TripleVectorial()
        t.m1 = 1e-3
        t.save_as_initial()
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['m1'], 1e-3)


class TestIntegration(unittest.TestCase):
    def test_integrate(self):
        """See that we can integrate the triple."""
        tv = TripleVectorial()
        ev = tv.evolve(1e3)
        self.assertGreater(len(ev), 0)

    def test_ecc_extrema(self):
        """See that we can use the eccmaxima function."""
        tv = TripleVectorial()
        ex = tv.extrema(1e4)
        self.assertGreater(len(ex), 0)

    def test_cputimeout(self):
        """Test that integration halts after exceeding the maximum CPU time."""
        large_time = 1e9
        tv = TripleVectorial()
        tv.cputstop = 0.1
        tv.evolve(large_time)
        self.assertLess(tv.t, large_time)

    def test_reset(self):
        """Try to reset the triple."""
        tv = TripleVectorial()
        tstop = 1e3
        tv.evolve(tstop)
        tv.reset()
        ev1 = tv.evolve(tstop)
        tv.reset()
        ev2 = tv.evolve(tstop)
        assert_allclose(ev1[-1], ev2[-1], rtol=1e-2, atol=1e-2)

    def test_flip_period(self):
        """Test the flip_period method."""
        tv = TripleVectorial(inc=80, e1=0.1, m1=1)
        p = tv.flip_period(nflips=3)
        assert_allclose(p, 98590, atol=10)
