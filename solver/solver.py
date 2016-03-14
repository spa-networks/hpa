# -*- coding: utf-8 -*-
# @author: Jean-Gabriel Young <jean.gabriel.young@gmail.com>

# Todo/method wishlist:
# - Safer interface
# - remove side effects from roll in get_sb, get_sg, get_q_prime.
"""Solver for the HPA equations."""
import numpy as np


def update_preferential_attachment(state_vec,
                                   birth_rate, growth_rate,
                                   t, offset=1):
    r"""
    Update a state vector using the generalized PA equation.

    The generalized PA equation allows one to track the number of entities in
    an average system following Simon's variant of preferential attachment.
    It is fully parametrized by a birth rate $B$, a growth rate $G$ and the
    time $t$ of the process (the latter allows an indirect and efficient
    computation of the average system size).

    Given a state vector $N_k(t)$ that counts the number of elements with $k$
    shares of the total resource $K(t)=t + K_0$ at time $t$, one obtains the
    average state vector at time $t+1$ from the generalized PA equation
    $N_k(t+1) = N_k(t) + B\delta_{k,1} + \frac{G}{(B+G)t + K_0} \times
    [(k-1)N_{k-1}(t) - k N_k(t)]$.

    Parameters
    ----------
    state_vec : np.array
      Unormalized distribution of resources at time $t$.
    birth_rate : float
      Rate of birth of entities.
    growth_rate : float
      Rate of growth of entities.
    t : int
      Time of the process.
    offset : int
      Initial size of the system.

    Returns
    -------
    updated_state_vec : np.array
      Unormalized distribution of resources at time $t$.

    Notes
    -----
    Will return a longer state vector if the distribution grows beyond the
    length of the input state vector.
    """
    if state_vec[-1] > 0:
        state_vec = np.concatenate((state_vec, [0]))
    dim = len(state_vec)
    return (state_vec +
            np.concatenate(([0], [birth_rate], np.zeros(dim - 2))) +
            growth_rate / ((birth_rate + growth_rate) * t + offset) *
            (np.roll(state_vec, 1) * (range(dim) - np.ones(dim)) -
             state_vec * range(dim)))


update = update_preferential_attachment  # alias


def get_sb(p):
    r"""
    Compute the rate of structural birth at all levels.

    The rate of structural birth at level k is given by

    $^{(S)}B_k=\sum_{i=1}^kp_i\prod_{j=1}^{i-1}(1-p_j)$

    with the convention that $\prod_{k=i}^j a_k =1$ and $\sum_{k=i}^j a_k = 0$
    when $j<i$.

    Parameters
    ----------
    p : np.array
      Probability of creating a new k-structure, for $k=0,...,d+1$.
      Note that one must have $p_0 = 0$ and $p_{d+1} = 1$.

    Returns
    -------
    sb : np.array
      Rate of structural birth at level all levels
    """
    return np.cumsum(p * np.roll(np.cumprod(np.ones(p.shape) - p), 1))


def get_sg(p):
    r"""
    Compute the rate of structural growth at all levels.

    The rate of structural growth at level k is given by

    $^{(S)}G_k=p_{k+1}\prod_{i=1}^{k}(1-p_i)$

    with the convention that $\prod_{k=i}^j a_k =1$.

    Parameters
    ----------
    p : np.array
      Probability of creating a new k-structure, for $k=0,...,d+1$.
      Note that one must have $p_0 = 0$ and $p_{d+1} = 1$.

    Returns
    -------
    sg : np.array
      Rate of structural growth at level all levels.
    """
    return np.roll(p, -1) * np.cumprod(np.ones(p.shape) - p)


def get_q_prime(q, sb, sg):
    r"""
    Compute the corrected probabilities that a node is new at all level.

    The corrected probability is given approximately by
    $q_{k}' = q_k + \frac{q_{k+1}}{1 + 2 ^{(S)}G_k / ^{(S)}B_k}$.

    Parameters
    ----------
    q : np.array
      Probability of choosing a new node for the selected structures.
      Note that one must have $q_{d} = 1$ and $q_{d+1}=0$.
    sg : np.array
      Rate of structural birth at level k, for k=0,...,d+1.
    sg : np.array
      Rate of structural growth at level k, for k=0,...,d+1.

    Returns
    -------
    q_prime : np.array
      Corrected probabilities that a node is new at all level.

    See Also
    --------
    get_sb, get_sg
    """
    with np.errstate(divide='ignore'):  # ignore sb[0] = 0
        return q + (np.roll(q, -1)) / (1 + 2 * sg / sb)


def get_r(q_prime, p):
    r"""
    Compute the stopping probabilities at all level.

    The stopping probability is the probability that the construction process
    ends by choosing an existing node at level $k$, considering $d$ levels of
    organization. It is defined recursively by $R_k(d) = (1-q'_{k})
    \left[p_{k+1}\prod_{i=0}^k(1-p_i) + \frac{q'_{k+1}R_{k+1}(d)}{1-q'_{k+1}}
    \right]$ with the start condition $R_{d-1}(d) = (1 - q'_{d-1})
    \prod_{i=0}^{d-1}(1-p_i)$.

    We use the convention that $\prod_{k=i}^j a_k =1$ if $j<i$.

    Parameters
    ----------
    q_prime : np.array
      Corrected probabilities that a node is new at all level.
    p : np.array
      Probability of creating a new k-structure, for $k=0,...,d+1$.
      Note that one must have $p_0 = 0$ and $p_{d+1} = 1$.

    Returns
    -------
    r : np.array
      Stopping probabilities at all level.

    See Also
    --------
    get_q_prime
    """
    d = len(p) - 2
    r = np.zeros_like(p)
    r[d - 1] = (1 - q_prime[d - 1]) * np.prod((np.ones_like(p) - p)[:d])
    for k in reversed(range(d - 1)):
        r[k] = (p[k + 1] * np.prod((np.ones_like(p) - p)[:k + 1]) +
                q_prime[k + 1] * r[k + 1] / (1 - q_prime[k + 1]))
        r[k] *= (1 - q_prime[k])
    return r


def get_nb(r, q):
    r"""
    Compute the rate of nodal birth at all levels.

    The rate of nodal birth at level k is given by

    $^{(N)}B_k= 1 - \frac{q_0}{1-q_0}R_0(d)$

    Parameters
    ----------
    r : np.array
      Stopping probabilities at all level.
    q : np.array
      Probability of choosing a new node for the selected structures.
      Note that one must have $q_{d} = 1$ and $q_{d+1}=0$.

    Returns
    -------
    nb : np.array
      Rate of nodal birth at level all levels

    See Also
    --------
    get_r
    """
    return q[0] * r[0] / (1 - q[0]) * np.ones_like(q)


def get_ng(r):
    r"""
    Compute the rate of nodal growth at all levels.

    The rate of nodal growth at level k is given by

    $^{(N)}G_k= \sum_{i=0}^{k-1}R_i(d)$

    with the convention that $\sum_{k=i}^j a_k = 0$ when $j<i$.

    Parameters
    ----------
    r : np.array
      Stopping probabilities at all level.
    q : np.array
      Probability of choosing a new node for the selected structures.
      Note that one must have $q_{d} = 1$ and $q_{d+1}=0$.

    Returns
    -------
    ng : np.array
      Rate of nodal growth at level all levels

    See Also
    --------
    get_r
    """
    return np.cumsum(np.roll(r, 1))


def get_all_rates(p, q):  # helper_function
    r"""
    Compute all the growth and birth rate vectors from a p,q vector pairs.

    Parameters
    ----------
    p : np.array
      Probability of creating a new k-structure, for $k=0,...,d+1$.
      Note that one must have $p_0 = 0$ and $p_{d+1} = 1$.
    q : np.array
      Probability of choosing a new node for the selected structures.
      Note that one must have $q_{d} = 1$ and $q_{d+1}=0$.

    Returns
    -------
    (sb, sg, ng, nb) : tuple of floats
      All growth rates.

    See Also
    --------
    get_sb, get_sh, get_ng, get_nb
    """
    sb = get_sb(p)
    sg = get_sg(p)
    q_prime = get_q_prime(q, sb, sg)
    r = get_r(q_prime, p)
    ng = get_ng(r)
    nb = get_nb(r, q)
    return sb, sg, ng, nb


def solve_hpa(p, q, t_max=1000):  # helper_function
    r"""
    Obtain the state vectors of a lenght t_max HPA process.

    Parameters
    ----------
    p : np.array
      Probability of creating a new k-structure, for $k=0,...,d+1$.
      Note that one must have $p_0 = 0$ and $p_{d+1} = 1$.
    q : np.array
      Probability of choosing a new node for the selected structures.
      Note that one must have $q_{d} = 1$ and $q_{d+1}=0$.
    t_max : int
      Number of iterations.

    Returns
    -------
    n : list of np.array
      Unormalized distributions of memberships at all levels.
    s : list of np.array
      Unormalized distributions of sizes at all levels.
    """
    d = len(p) - 2
    (sb, sg, ng, nb) = get_all_rates(p, q)
    init = np.array([0, 1])
    n = [init.copy() for i in range(d)]
    s = [init.copy() for i in range(d)]
    for t in range(1, t_max):
        for k in range(1, d + 1):
            s[k - 1] = update(s[k - 1], sb[k], sg[k], t)
            n[k - 1] = update(n[k - 1], nb[k], ng[k], t)
    return n, s


if __name__ == "__main__":
    # Options parser.
    import argparse as ap
    import sys
    prs = ap.ArgumentParser(description="Solver of HPA's equations for size\
                                         and membership distributions.")
    prs.add_argument('--p', '-p', type=float, nargs='+', required=True,
                     help='Vector of p_k.')
    prs.add_argument('--q', '-q', type=float, nargs='+', required=True,
                     help='Vector of q_k.')
    prs.add_argument('--t', '-t', type=int, default=100,
                     help='Iteration time.')
    prs.add_argument('--k', '-k', type=int, required=True,
                     help='Level of the distribution.')
    prs.add_argument('--type', type=str, default='size',
                     choices=['size', 'membership'],
                     help='Type of distribution.')
    if len(sys.argv) == 1:
        prs.print_help()
        sys.exit(1)
    args = prs.parse_args()
    p = np.array(args.p)
    q = np.array(args.q)

    # Determine all nodal and structural birth / growth rates from p,q.
    (sb, sg, ng, nb) = get_all_rates(p, q)

    # Initialize and run
    state = np.array([0, 1])  # one element with one resource share
    if args.type == 'size':
        birth = sb[args.k]
        growth = sg[args.k]
    if args.type == 'membership':
        birth = nb[args.k]
        growth = ng[args.k]
    for t in range(1, args.t):
        state = update_preferential_attachment(state, birth, growth, t)

    # Output
    normalization = sum(state)
    for i, val in enumerate(state):
        print(str(i).ljust(3), "\t", val / normalization)
