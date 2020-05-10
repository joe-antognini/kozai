import json
import unittest

from numpy.testing import assert_allclose

from kozai.delaunay import TripleDelaunay


class TestObjectCreation(unittest.TestCase):
    def test_make_Triple(self):
        '''Try to create a Triple class.'''
        t = TripleDelaunay()
        self.assertIsNotNone(t)

    def test_nooct(self):
        '''Try to turn the octupole term off.'''
        t = TripleDelaunay()
        t.octupole = False
        self.assertFalse(t.octupole)

    def test_setoptions(self):
        '''Try to set some options.'''
        t = TripleDelaunay(
            a1=1,
            a2=20,
            e1=0.1,
            e2=0.3,
            m1=1,
            m2=0.5,
            m3=2,
            g1=90,
            g2=25,
            r1=1,
            r2=2,
            inc=70,
        )
        assert_allclose(t.inc, 70)

    def test_set_after_creation(self):
        '''Try to set a parameter after the object has been created.'''
        t = TripleDelaunay(m1=1, m2=1, m3=1, inc=80)
        t.m2 = 1e-3
        assert_allclose(t.inc, 80)


class TestParameterCalculation(unittest.TestCase):
    def test_Th(self):
        '''Test the calculation of Kozai's integral.'''
        t = TripleDelaunay(e1=0.05, inc=70)
        assert_allclose(t.Th, 0.11668533399441)

    def test_epsoct(self):
        '''Test the epsoct calculation.'''
        t = TripleDelaunay(a1=1, a2=20, e2=0.3)
        assert_allclose(t.epsoct, 0.01648351648)

    def test_CKL(self):
        '''Test the CKL calculation.'''
        t = TripleDelaunay(e1=0.1, inc=80, g1=45)
        assert_allclose(t.CKL, -0.00212307888)

    def test_Hhatquad(self):
        t = TripleDelaunay(e1=0.1, inc=80, g1=45)
        assert_allclose(t.Hhatquad, 1.846364030293)


class TestRepresentation(unittest.TestCase):
    def test_repr(self):
        '''Make sure that the triple can print its state.'''
        t = TripleDelaunay(inc=80, m2=1e-3)
        t.octupole = False
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['m2'], 1e-3)
        assert_allclose(state['inc'], 80)
        self.assertFalse(state['octupole'])

    def test_save_as_initial(self):
        '''Try to set a parameter after creating the object and saving it as an
        initial condition.'''
        t = TripleDelaunay()
        t.m2 = 1e-3
        t.save_as_initial()
        j = t.__repr__()
        state = json.loads(j)
        assert_allclose(state['m2'], 1e-3)


class TestIntegration(unittest.TestCase):
    def test_evolve(self):
        '''Try to integrate a triple.'''
        t = TripleDelaunay()
        ev = t.evolve(1e3)
        self.assertGreater(len(ev), 0)

    def test_extrema(self):
        '''See that we can use the eccmaxima function.'''
        t = TripleDelaunay()
        ev = t.extrema(1e4)
        self.assertGreater(len(ev), 0)

    def test_cputimeout(self):
        '''Make sure that the integration halts after exceeding the maximum CPU
        integration time.'''
        large_time = 1e9
        t = TripleDelaunay()
        t.cputstop = 0.1
        t.evolve(large_time)
        self.assertLess(t.t, large_time)

    def test_find_flip(self):
        '''Make sure that flips are being found.'''
        t = TripleDelaunay()
        t.m2 = 1e-3
        flips = t.find_flips(1e5)
        self.assertGreater(len(flips), 0)

    def test_reset(self):
        '''Try to reset the triple.'''
        t = TripleDelaunay()
        tstop = 1e3
        t.evolve(tstop)
        t.reset()
        ev1 = t.evolve(tstop)
        t.reset()
        ev2 = t.evolve(tstop)
        assert_allclose(ev1[-1], ev2[-1], rtol=1e-2, atol=1e-2)

    def test_collision(self):
        '''Make sure that collisions happen (or not).'''
        t = TripleDelaunay(r1=1e-3, r2=1e-3)
        t.evolve(1e4)
        self.assertFalse(t.collision)

        t = TripleDelaunay(r1=30, r2=30)
        t.evolve(1e4)
        self.assertTrue(t.collision)


# TODO: Test conservation of constants.
