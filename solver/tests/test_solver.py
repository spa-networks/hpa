#!/usr/bin/env python
"""Unit tests for HPA utilities, using nosetest."""
from nose.tools import assert_almost_equals, assert_equal
from solver import solver
import numpy as np

# Default test data.
p = np.array([0, 0.0005, 0.185, 0.385, 1])
q = np.array([0.80, 0.60, 0.50, 1, 0])


class TestSolver:
    """Unit tests for the HPA solver."""

    def test_update_preferential_attachment(self):
        """[solver.py] Unit test for update_preferential_attachment."""
        update = solver.update_preferential_attachment
        # test vector length growth
        assert_equal(3, len(update(np.array([0, 1]), 1, 1, 1)))
        assert_equal(3, len(update(np.array([0, 1, 0]), 1, 1, 1)))

        # test updates for multiple parameters
        def manual_update(b, g, t):
            return np.array([0,
                             1 + b - g / ((b + g) * t + 1),
                             g / ((b + g) * t + 1)])

        for i in range(len(p)):
            for j in range(len(q)):
                for t in range(1, 5):
                    x = update(np.array([0, 1, 0]), p[i], q[j], t)
                    y = manual_update(p[i], q[j], t)
                    for idx in range(3):
                        assert_almost_equals(x[idx], y[idx])

    def test_get_sb(self):
        """[solver.py] Unit test for get_sb."""
        # Implements the equations of Table II of PRE 92, 068809 (2015).
        sb = solver.get_sb(p)
        assert_almost_equals(sb[0], p[0])
        assert_almost_equals(sb[1], p[1])
        assert_almost_equals(sb[2], p[1] + p[2] * (1 - p[1]))
        assert_almost_equals(sb[3], p[1] + p[2] * (1 - p[1]) +
                             p[3] * (1 - p[1]) * (1 - p[2]))

    def test_get_sg(self):
        """[solver.py] Unit test for get_sg."""
        # Implements the equations of Table II of PRE 92, 068809 (2015).
        sg = solver.get_sg(p)
        assert_almost_equals(sg[0], p[1])
        assert_almost_equals(sg[1], p[2] * (1 - p[1]))
        assert_almost_equals(sg[2], p[3] * (1 - p[1]) * (1 - p[2]))
        assert_almost_equals(sg[3], (1 - p[1]) * (1 - p[2]) * (1 - p[3]))

    def test_get_q_prime(self):
        """[solver.py] Unit test for get_q_prime."""
        sb = solver.get_sb(p)
        sg = solver.get_sg(p)
        qprime = solver.get_q_prime(q, sb, sg)

        def explicit_q_k_prime(k, q, sb, sg):
            with np.errstate(divide='ignore'):
                return q[k] + q[k + 1] / (1 + 2 * sg[k] / sb[k])

        assert_almost_equals(qprime[0], explicit_q_k_prime(0, q, sb, sg))
        assert_almost_equals(qprime[1], explicit_q_k_prime(1, q, sb, sg))
        assert_almost_equals(qprime[2], explicit_q_k_prime(2, q, sb, sg))

    def test_get_r(self):
        """[solver.py] Unit test for get_r."""
        # Implements the equations of Table II of PRE 92, 068809 (2015).
        sb = solver.get_sb(p)
        sg = solver.get_sg(p)
        q_prime = solver.get_q_prime(q, sb, sg)
        r = solver.get_r(q_prime, p)
        assert_almost_equals(r[0], (1 - q[0]) * (p[1] + q_prime[1] * (1 - p[1]) * (p[2] + q_prime[2] * (1 - p[2])))) # noqa
        assert_almost_equals(r[1], (1 - q_prime[1]) * (1 - p[1]) * (p[2] + q_prime[2] * (1 - p[2]))) # noqa
        assert_almost_equals(r[2], (1 - q_prime[2]) * (1 - p[1]) * (1 - p[2]))

    def test_get_nb(self):
        """[solver.py] Unit test for get_nb."""
        # Implements the equations of Table II of PRE 92, 068809 (2015).
        sb = solver.get_sb(p)
        sg = solver.get_sg(p)
        q_prime = solver.get_q_prime(q, sb, sg)
        r = solver.get_r(q_prime, p)
        nb = solver.get_nb(r, q)
        nb_all = q[0] * (p[1] + q_prime[1] * (1 - p[1]) * (p[2] + q_prime[2] * (1 - p[2]))) # noqa
        assert_almost_equals(nb[0], nb_all)
        assert_almost_equals(nb[1], nb_all)
        assert_almost_equals(nb[2], nb_all)
        assert_almost_equals(nb[3], nb_all)

    def test_get_ng(self):
        """[solver.py] Unit test for get_ng."""
        # Implements the equations of Table II of PRE 92, 068809 (2015).
        sb = solver.get_sb(p)
        sg = solver.get_sg(p)
        q_prime = solver.get_q_prime(q, sb, sg)
        r = solver.get_r(q_prime, p)
        ng = solver.get_ng(r)
        assert_almost_equals(ng[0], 0)
        assert_almost_equals(ng[1], r[0])
        assert_almost_equals(ng[2], r[0] + r[1])
        assert_almost_equals(ng[3], r[0] + r[1] + r[2])
