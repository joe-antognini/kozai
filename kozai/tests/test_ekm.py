import json
import unittest

from numpy.testing import assert_allclose

from kozai.ekm import F
from kozai.ekm import TripleOctupole


class TestFunctions(unittest.TestCase):
    def test_F(self):
        """Test the calculation of F."""
        assert_allclose(F(0.015), 0.017068821850895335)


class TestObjectCreation(unittest.TestCase):
    def test_triple_octupole(self):
        """Try to create a Triple_octupole class."""
        to = TripleOctupole()
        self.assertIsNotNone(to)

    def test_set_options_delaunay(self):
        """See if we can set some options."""
        to = TripleOctupole(e1=0.05, inc=85, Omega=90, g1=25)
        self.assertIsNotNone(to)

    def test_set_options_constants(self):
        """Set the integrals of motion"""
        to = TripleOctupole(phiq=0.15, chi=F(0.15), epsoct=0.01)
        self.assertIsNotNone(to)


class TestRepresentation(unittest.TestCase):
    def test_repr(self):
        """Make sure that the triple can print its state."""
        t = TripleOctupole(inc=70, Omega=170)
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['inc'], 70)
        assert_allclose(state['Omega'], 170)

    def test_save_as_initial(self):
        """Try to set a parameter after creating the object and saving it as an
        initial condition."""
        t = TripleOctupole()
        t.phiq = 0.015
        t.save_as_initial()
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['phiq'], 0.015)


class TestIntegration(unittest.TestCase):
    def test_evolve(self):
        """Integrate the object."""
        to = TripleOctupole()
        to.evolve(10)
        self.assertIsNotNone(to)

    def test_const_evolve(self):
        """Integrate an object that was set with the integrals of motion."""
        to = TripleOctupole(phiq=0.15, chi=F(0.15), epsoct=0.01)
        to.evolve(10)
        self.assertIsNotNone(to)

    def test_cputimeout(self):
        """Make sure that the integration halts after exceeding the maximum CPU
        integration time."""
        large_time = 1e9
        to = TripleOctupole()
        to.cputstop = 0.1
        to.evolve(large_time)
        self.assertLess(to.t, large_time)

    def test_reset(self):
        """Try to reset the triple."""
        t = TripleOctupole()
        tstop = 10
        t.evolve(tstop)
        t.reset()
        ev1 = t.evolve(tstop)
        t.reset()
        ev2 = t.evolve(tstop)
        assert_allclose(ev1[-1], ev2[-1], rtol=1e-2, atol=1e-2)

    def test_flip_period(self):
        """Test the analytic flip period calculation"""
        t = TripleOctupole(e1=0.1, inc=80, Omega=180, g1=0, epsoct=1e-2)
        p = t.flip_period()
        assert_allclose(p, 123.6079642436734, atol=1e-3, rtol=1e-3)

    def test_numeric_flip_period(self):
        """Test the numeric calculation of the flip period"""
        t = TripleOctupole(e1=0.1, inc=80, Omega=180, g1=0, epsoct=1e-2)
        p = t.numeric_flip_period(n_flips=3)
        assert_allclose(p, 123.58242387832222, atol=1e-3, rtol=1e-3)
