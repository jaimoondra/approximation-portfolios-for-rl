import numpy as np


def generalized_p_mean(x, p):
    """
    Generalized p-mean function, defined as follows:
    f(x) = (1/d * sum(x_i^p))^(1/p)
    where x is a vector of d elements.
    :param x: a strictly positive vector
    :param p: real number, or -np.inf, or np.inf
    :return: generalized p-mean of x
    """
    d = len(x)                                              # Dimension of the vector

    for i in range(d):                                      # Check if the vector is non-negative
        if x[i] < 0:
            raise ValueError('x must be non-negative')

    # If p <= 0 and there is a zero element in the vector, return 0
    if p <= 0:
        for i in range(d):
            if x[i] == 0:
                return 0

    if p == -np.inf:                                        # If p = -\infty, return the minimum value of the vector
        value = np.min(x)
    elif p == 0:                                            # If p = 0, return the geometric mean of the vector
        value = np.prod(x) ** (1/d)
    elif p == np.inf:                                       # If p = \infty, return the maximum value of the vector
        value = np.max(x)
    else:                                                   # Otherwise, return the generalized p-mean
        value = (np.sum([x[i] ** p for i in range(d)])/d) ** (1/p)

    return value


def line_search(current_vector, start_p, alpha, get_optimum, precision=1e-4, upper_bound=1, mu=0.5):
    """
    Given a current vector, a starting value for p, a parameter alpha \in (0, 1), a subroutine to find the optimum vector for any q, a precision value, an upper bound for p, and a weight mu for the line search, returns a value of p such that the current vector is an alpha-approximation for all generalized p-means between starting value of p and the returned value of p.
    :param current_vector: the candidate vector for the portfolio
    :param start_p: starting value of p, we assume that the vector is optimal for this value of p
    :param alpha: desired approximation factor
    :param get_optimum: subroutine to find the vector that maximizes the generalized p-mean function, should take a 'p' value as input and return a tuple (optimum_value, vector [np array])
    :param precision: used to stop the line search
    :param upper_bound: upper bound for p
    :param mu: weight for the line search, default is 0.5
    :return: a value of p such that current_vector is an alpha-approximation for all q in [start_p, returned value of p]
    """
    is_alpha_approximation = True                           # Flag to stop the line search

    lower_bound = start_p                                   # Initialize the lower bound
    p = mu * lower_bound + (1 - mu) * upper_bound           # Initialize the value of p
    value = generalized_p_mean(current_vector, start_p)     # Initialize the value of the generalized p-mean function

    while is_alpha_approximation:                           # Iterate until current_vector is not an alpha-approximation
        opt_p, vector = get_optimum(p)        # Find the vector that maximizes the generalized p-mean function

        # If the value of the function is at least alpha times the optimal value, then current_vector is an alpha-approximation for all q in [lower_bound, p]
        # Therefore, update the lower bound and the value of the function
        if value >= alpha * opt_p:
            lower_bound = p
            value = generalized_p_mean(current_vector, p)
        # Otherwise, update the upper bound
        else:
            upper_bound = p

        # Update the value of p using the line search
        p = mu * lower_bound + (1 - mu) * upper_bound
        print('\tnext search p: ', p)
        # If the difference between the upper and lower bounds is less than the precision, stop the line search
        if upper_bound - lower_bound < precision:
            is_alpha_approximation = False

    return upper_bound


def portfolio_with_line_search(alpha, get_optimum, mu=0.5):
    """
    Given desired approximation alpha \in (0, 1) and a subroutine to find the optimum for any p-mean, returns an alpha-approximate portfolio for the generalized p-mean functions, p \in [-\infty, 1]
    :param alpha: real number in (0, 1)
    :param get_optimum: subroutine to find the vector that maximizes the generalized p-mean function, should take a 'p' value as input and return a tuple (optimum_value, vector [np array])
    :param mu: any real number in (0, 1), weight for line search subroutine, default is 0.5. Leave unchanged if unsure
    :return: the portfolio that maximizes the generalized p-mean, and the value of the function
    """
    opt_min, vector_min = get_optimum(-np.inf)  # Find the vector that maximizes the generalized p-mean function for p = -\infty
    print(f'p: {-np.inf}, P mean: {opt_min}, vector: {vector_min}' )
    d = len(vector_min)                       # Dimension of the vector

    p = np.log(d)/np.log(alpha)             # Start with the value of p that approximates p = - \infty
    portfolio = set()                          # Initialize the portfolio
    
    while p < 1:                            # Iterate until p = 1
        # Find the vector that maximizes the generalized p-mean function
        opt_p, vector = get_optimum(p)
        print(f'p: {p}, P mean: {opt_p}, vector: {vector}' )
        # if tuple(vector) not in portfolio:
        #     # Add vector to the portfolio if it is not already there
        portfolio.add(tuple([p]+list(vector)))

        # Find the next value of p using line search
        if p<-20:
            ada_precision = 10
        elif p<-5:
            ada_precision = 3
        elif p<0:
            ada_precision = 1
        else:
            ada_precision = 0.3
        p = line_search(current_vector=vector, start_p=p, alpha=alpha, get_optimum=get_optimum, upper_bound=1, mu=mu)
        print('portfolio: ', portfolio)
        print('next p: ', p)

    return portfolio
