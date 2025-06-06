import numpy as np


def generalized_p_mean(x, p, epsilon=1e-12):
    """
    (Numerically stable) generalized p-mean function, defined as follows:
    f(x) = (1/d * sum(x_i^p))^(1/p)
    where x is a vector of d elements.
    :param x: a strictly positive vector
    :param p: real number, or -np.inf, or np.inf
    :param epsilon: multiplicative approximation error
    :return: generalized p-mean of x, up to factor (1 - epsilon)
    """
    d = len(x)                                              # Dimension of the vector
    y = np.array(x)

    if y.any() < 0:
        raise ValueError('x must be strictly positive')

    if p < 0:
        if np.min(y) == 0:
            return 0
        else:
            return 1/(generalized_p_mean(x=1/y, p=-p, epsilon=epsilon))

    if p == 0:                                              # If p = 0, return the geometric mean of the vector
        return np.prod(y) ** (1/d)

    # When p > 0:
    p_max = - np.log(d)/np.log(1 - epsilon)
    if p >= p_max:                                          # Return the value at inf if p >= p_max
        value = np.max(y)
    else:
        z = y/np.max(y)
        value = np.max(y) * np.power(np.mean(np.power(z, p)), 1/p)

    return value


def get_optimum_vector(vectors, p, epsilon=1e-12):
    """
    Given a p \in [-\infty, 1] and a list of vectors (each vector is a list of d elements), find the vector that maximizes the generalized p-mean.
    :param vectors: iterable of vectors
    :param p: real number <= 1 or -np.inf
    :return: the vector that maximizes the generalized p-mean, and the value of the function
    """
    max_value = -np.inf
    max_vector = None

    for i in range(len(vectors)):
        vector = vectors[i]
        value = generalized_p_mean(vector, p, epsilon=epsilon)
        if value > max_value:
            max_value = value
            max_vector = vector

    return max_vector


def get_optimum_value(vectors, p, epsilon=1e-12):
    """
    Given a p \in [-\infty, 1] and a list of vectors (each vector is a list of d elements), find the value of the generalized p-mean function that maximizes the function.
    :param vectors: iterable of vectors
    :param p: real number <= 1 or -np.inf
    :return: the value of the function that maximizes the generalized p-mean
    """
    max_value = -np.inf

    for i in range(len(vectors)):
        vector = vectors[i]
        value = generalized_p_mean(vector, p, epsilon=epsilon)
        if value > max_value:
            max_value = value

    return max_value


def generate_p_grid(N, alpha, grid_size=1000, p_mid=None):
    """
    Generates a grid of values of p between -\infty and 1.
    :param N: number of reward functions
    :param alpha: real number in (0, 1)
    :param grid_size: number of desired points in the grid
    :return: grid of values of p
    """
    grid = []

    if p_mid is None:
        p_mid = - np.log(N)/np.log(1/alpha)

    q_least = 0
    q_most = 1/p_mid

    subgrid_size = grid_size//2 - 1

    grid.append(-np.inf)
    for i in range(1, subgrid_size + 1):
        q = q_least + i * (q_most - q_least) / subgrid_size
        grid.append(1/q)

    p_most = 1
    for i in range(1, subgrid_size + 1):
        p = p_mid + i * (p_most - p_mid) / subgrid_size
        grid.append(p)

    grid.append(0)
    grid.sort()

    return grid
