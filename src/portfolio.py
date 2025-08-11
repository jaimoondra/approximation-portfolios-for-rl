from src.p_mean import generalized_p_mean, get_optimum_vector, get_optimum_value
import numpy as np
from scipy.optimize import differential_evolution


class Policy:
    def __init__(self, id, **attributes):
        self.id = id
        self.__dict__.update(attributes)

    def __repr__(self):
        attrs = ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if k != 'id')
        return f"Policy(id={self.id}, {attrs})"


class Portfolio:
    def __init__(self, name=""):
        self.name = name
        self.policies = []
        self.oracle_calls = 0

    def __len__(self):
        return len(self.policies)

    def add_policy(self, policy):
        self.policies.append(policy)

    def remove_policy(self, policy):
        self.policies.remove(policy)

    def sort_by(self, attribute, reverse=False):
        """
        Return policies sorted by the specified attribute.

        Args:
            attribute (str): The attribute name to sort by
            reverse (bool, optional): If True, sort in descending order. Default is False.

        Returns:
            list: Sorted list of policies

        Raises:
            AttributeError: If any policy doesn't have the specified attribute
        """
        # Check if all policies have the attribute to avoid unexpected errors during sorting
        if not all(hasattr(policy, attribute) for policy in self.policies):
            raise AttributeError(f"Not all policies have the attribute '{attribute}'")

        return sorted(self.policies, key=lambda policy: getattr(policy, attribute), reverse=reverse)

    def __iter__(self):
        return iter(self.policies)

    def __len__(self):
        return len(self.policies)

    def __repr__(self):
        return f"Portfolio(name={self.name}, policies={self.policies})"

    def set_oracle_calls(self, oracle_calls):
        self.oracle_calls = oracle_calls


def budget_portfolio_with_suboptimalities(initial_p, K, get_optimum_policy, get_performance):
    """
    Build a budget-constrained portfolio up to size K.
    initial p is the smallest p you start with.
    Returns:
      portfolio   : list of (p_value, policy) for each step
    """
    portfolio = Portfolio('Heuristic portfolio')

    for t in range(1, K+1):
        if t == 1:
            p_t = initial_p
            policy_t = Policy(get_optimum_policy(p_t), p=p_t)
            portfolio.add_policy(policy_t)
            # portfolio.append((p_t, policy_t))
        elif t == 2:
            p_t = 1.0
            policy_t = Policy(get_optimum_policy(p_t), p=p_t)
            portfolio.add_policy(policy_t)
            # portfolio.append((p_t, policy_t))
        else:
            # choose the interval with the smallest ratio (i.e., worst suboptimality)
            # portfolio.sort(key=lambda x: x[0])
            sorted_portfolio = portfolio.sort_by(attribute='p')
            min_ratio = 1
            chosen_interval = None

            for i in range(len(portfolio) - 1):
                policy_left, p_left = sorted_portfolio[i], sorted_portfolio[i].p
                policy_right, p_right = sorted_portfolio[i + 1], sorted_portfolio[i + 1].p
                # p_right, policy_right = sorted_portfolio[i + 1]

                perf_right_right = get_performance(policy_right.id, p_right) #optimal value for the p in the right side of the interval
                perf_left_right = get_performance(policy_left.id, p_right) #value of the policy in the left side of the interval evaluted w.r.t p in the right side of the interval
                if perf_right_right <= 1e-12:
                    print("Score too small. Numerical instability")
                    continue

                ratio = perf_left_right / perf_right_right
                if ratio < min_ratio:
                    min_ratio = ratio
                    chosen_interval = (p_left, p_right)

            if chosen_interval is None:
                # fallback
                print("Error. No next interval Chosen")
                p_t = 0.0
            else:
                p_t = 0.5 * (chosen_interval[0] + chosen_interval[1])

            policy_t = Policy(get_optimum_policy(p_t), p=p_t)
            portfolio.add_policy(policy_t)

    portfolio.set_oracle_calls(K)
    return portfolio


def line_search(get_optimum_policy, get_performance, current_policy, start_p, alpha, precision=1e-4, upper_bound=1, mu=0.5, variant=1):
    """
    Given a list of vectors, a current vector, a starting value for p, a parameter alpha \in (0, 1), a precision value, an upper bound for p, and a weight mu for the line search, returns a value of p such that the current vector is an alpha-approximation for all generalized p-means between starting value of p and the returned value of p.
    :param vectors: array of vectors in the domain (e.g., each vector is a score vector for a reward function)
    :param current_vector: the candidate vector for the portfolio, replace with a policy object if needed
    :param start_p: starting value of p, we assume that the vector is optimal for this value of p
    :param alpha: desired approximation factor
    :param precision: used to stop the line search
    :param upper_bound: upper bound for p
    :param mu: weight for the line search, default is 0.5
    :return: a value of p such that current_vector is an alpha-approximation for all q in [start_p, returned value of p]
    """
    lower_bound = start_p                                   # Initialize the lower bound
    number_of_oracle_calls = 0

    policy = get_optimum_policy(upper_bound)
    opt_upper = get_performance(policy, upper_bound)  # Find the value of the generalized p-mean function that maximizes the function, replace with your function if needed
    # opt_upper = get_optimum_value(vectors, upper_bound)        # Find the vector that maximizes the generalized p-mean function, replace with your function if needed
    number_of_oracle_calls = number_of_oracle_calls + 1

    while (generalized_p_mean(current_policy, lower_bound) < alpha * opt_upper) and (upper_bound - lower_bound > precision):                           # Iterate until current_vector is not an alpha-approximation
        q = mu * lower_bound + (1 - mu) * upper_bound
        policy_q = get_optimum_policy(q)          # Get the optimal policy for the current value of p
        opt_q = get_performance(policy_q, q)  # Find the value of the generalized p-mean function that maximizes the function, replace with your function if needed
        # opt_q = get_optimum_value(vectors, q)        # Find the vector that maximizes the generalized p-mean function, replace with your function if needed
        number_of_oracle_calls = number_of_oracle_calls + 1

        # If the value of the function is at least alpha times the optimal value, then current_vector is an alpha-approximation for all q in [lower_bound, p]
        # Therefore, update the lower bound and the value of the function
        threshold = (alpha + precision * (1 - alpha)) if variant == 1 else np.sqrt(alpha)
        if generalized_p_mean(current_policy, lower_bound) >= threshold * opt_q:
            lower_bound = q
        # Otherwise, update the upper bound
        else:
            upper_bound = q
            policy_upper = get_optimum_policy(upper_bound)  # Get the optimal policy for the new upper bound
            opt_upper = get_performance(policy_upper, upper_bound)  # Find the value of the generalized p-mean function that maximizes the function, replace with your function if needed
            # opt_upper = get_optimum_value(vectors, upper_bound)

    return upper_bound, number_of_oracle_calls


def portfolio_with_line_search(get_optimum_policy, get_performance, d, alpha, mu=0.5, variant=1):
    """
    Given a list of vectors and a parameter alpha \in (0, 1), returns an alpha-approximate portfolio for the generalized p-mean functions, p \in [-\infty, 1].
    :param vectors: iterable of vectors
    :param alpha: real number in (0, 1)
    :param mu: any real number in (0, 1), weight for line search subroutine, default is 0.5
    :param variant: variant of the line search algorithm, 1 is the original version, 2 is the modified version that uses a different stopping criterion
    :return: the portfolio that maximizes the generalized p-mean, and the value of the function
    """
    portfolio = Portfolio('p-Mean Portfolio')

    p = np.log(d)/np.log(alpha)             # Start with the value of p that approximates p = - \infty
    # vector = get_optimum_vector(vectors, p)  # Find the vector that maximizes the generalized p-mean function
    vector = get_optimum_policy(p)          # Get the optimal policy for the initial p
    portfolio.add_policy(Policy(vector, p=p))
    prev_p = p
    number_of_oracle_calls = 0

    while p < 1:                            # Iterate until p = 1
        # Find the next value of p using line search
        # p, iteration_oracle_calls = line_search(vectors=vectors, current_vector=vector, start_p=p, alpha=alpha, upper_bound=1, mu=mu, variant=variant)
        p, iteration_oracle_calls = line_search(
            get_optimum_policy=get_optimum_policy,
            get_performance=get_performance,
            current_policy=vector,
            start_p=p, alpha=alpha, upper_bound=1, mu=mu, variant=variant
        )

        # Check if the vector is already in the portfolio
        if not any(tuple(policy.id) == tuple(vector) for policy in portfolio):
            # Add vector to the portfolio if it is not already there
            portfolio.add_policy(Policy(vector, p=prev_p))

        # vector = get_optimum_vector(vectors, p)
        vector = get_optimum_policy(p)      # Get the optimal policy for the new p
        prev_p = p
        number_of_oracle_calls = number_of_oracle_calls + iteration_oracle_calls

    portfolio.set_oracle_calls(number_of_oracle_calls)

    return portfolio


def compute_portfolio_worst_approx_ratio(
        portfolio,  # list of policies (or policy vectors)
        get_optimal_value,         # function that takes in p and returns optimal value for that p
        p_grid,              # sorted list of p-values
        get_performance
):
    """
    Given:
      - portfolio_policies: a list of policies
      - p_to_optval: precomputed optimal performance for each p in p_grid
      - p_grid: list of p-values spanning [initial_p, 1]
      - get_performance function

    We compute, for each p in p_grid:
       ratio_p = max_{pi in portfolio} [ v^{(l)}(pi,p)  / v^{(l)}(globally optimal policy,p)  ]

    Then the 'worst-case approximation ratio' = min_{p in p_grid} ratio_p.
    """
    worst_ratio = float('inf')

    for p_val in p_grid:
        opt_val = get_optimal_value(p_val)

        # If the optimal is near zero, you have to decide how to handle that (skip or set ratio=1 or =∞).
        if opt_val <= 1e-12:
            # For demonstration, we skip p_val where optimum ~ 0
            # (or you might define ratio = 1 if your policy is also 0, or ∞ if your policy is > 0)
            continue

        # 1) Evaluate each policy in the portfolio
        best_portfolio_val = 0.0
        for policy in portfolio.policies:
            # print(policy)
            val = get_performance(policy.id, p_val)
            if val > best_portfolio_val:
                best_portfolio_val = val

        # 2) Compute the ratio for p_val
        ratio_p = best_portfolio_val / opt_val

        # 3) Track the minimum across all p
        if ratio_p < worst_ratio:
            worst_ratio = ratio_p

    return worst_ratio


def portfolio_of_random_policies(policies, K, seed=None):
    """
    Build a random portfolio up to size K.
    Returns:
      portfolio   : list of (p_value, policy) for each step
    """

    if seed:
        np.random.seed(seed)

    random_indices = np.random.choice(len(policies), replace=False, size=K)
    portfolio = Portfolio('Random Policy Portfolio')
    for i in random_indices:
        policy = policies[i]
        portfolio.add_policy(policy)

    return portfolio


def portfolio_of_random_norms(initial_p, K, get_optimum_policy, seed=None):
    """
    Build a random portfolio up to size K.
    initial p is the smallest p you start with.
    Returns:
      portfolio   : list of (p_value, policy) for each step
    """

    if seed:
        np.random.seed(seed)

    portfolio = Portfolio('Random Norm Portfolio')

    for t in range(1, K+1):
        p_t = np.random.uniform(initial_p, 1)
        policy_t = Policy(get_optimum_policy(p_t), p=p_t)
        portfolio.add_policy(policy_t)

    # portfolio.sort(key=lambda x: x[0])
    return portfolio


def weighted_average(x, weight_vector):
    """
    Calculate the weighted average of a list of values.

    Parameters:
    x (np.array | tuple | List): Array of values.
    weight_vector (np.array | tuple | List): Array of weights.

    Returns:
    float: Weighted average.
    """
    return np.dot(np.array(x), np.array(weight_vector))


def optimal_vector_for_given_weight(vectors, weight_vector):
    """
    Given a weight vector and a list of vectors, find the vector that maximizes the weighted average.

    Parameters:
    vectors (np.array | tuple | List): Array of vectors.
    weight_vector (np.array | tuple | List): Array of weights.

    Returns:
    np.array: Optimal vector.
    """
    optimal_vector = None
    optimal_value = - np.inf
    for vector in vectors:
        value = weighted_average(vector, weight_vector)
        if value > optimal_value:
            optimal_value = value
            optimal_vector = vector

    return optimal_vector, optimal_value


def find_maximal_weight_vector_evolution(vectors, chosen_vectors):
    d = len(vectors[0])
    vectors = np.array(vectors)
    chosen_vectors = np.array(chosen_vectors)

    def objective(w_raw):
        w = np.maximum(w_raw, 0)
        w /= np.sum(w) + 1e-12  # Normalize to ensure sum to 1 and avoid divide-by-zero
        opt_all = np.max(vectors @ w)
        opt_chosen = np.max(chosen_vectors @ w)
        return opt_chosen - opt_all  # want to minimize this difference

    bounds = [(0, 1)] * d
    result = differential_evolution(objective, bounds, tol=1e-2, popsize=100, maxiter=250, disp=True)

    w_opt = result.x / np.sum(result.x)
    scores = vectors @ w_opt
    i_star = np.argmax(scores)
    chosen_v = vectors[i_star]
    obj_val = np.max(scores) - np.max(chosen_vectors @ w_opt)
    return w_opt.tolist(), chosen_v.tolist(), obj_val


def gpi(vectors, portfolio_size=10):
    weight_vectors = []
    chosen_vectors = []

    N = len(vectors[0]) if len(vectors) > 0 else 0

    weight_vector = np.array([1] + [0] * (N - 1))  # Start with the first vector having weight 1 and others 0
    optimal_vector, optimal_value = optimal_vector_for_given_weight(vectors, weight_vector)

    # Store the first vector as the chosen vector
    chosen_vectors.append(optimal_vector)
    weight_vectors.append(weight_vector)

    # Iterate to find the next vectors
    iteration = 0
    while optimal_value > 0:
        print(f"Iteration {iteration}")
        iteration += 1
        # Find the next weight vector that maximizes the objective function
        # based on the current chosen vectors
        weight_vector, optimal_vector, optimal_value = find_maximal_weight_vector_evolution(vectors, chosen_vectors)

        if optimal_vector is None:
            raise ValueError("No optimal vector found. The optimization process failed.")

        # Append the chosen vector to the list of chosen vectors
        chosen_vectors.append(optimal_vector)
        # Append the weight vector to the list of weight vectors
        weight_vectors.append(weight_vector)

        if iteration >= portfolio_size:
            print("Stopping after 10 iterations.")
            break

    return weight_vectors, chosen_vectors


def portfolio_with_gpi(vectors, portfolio_size=10):
    weight_vectors, chosen_vectors = gpi(vectors, portfolio_size)

    portfolio = Portfolio('GPI Portfolio')
    for chosen_vector in chosen_vectors:
        policy = Policy(chosen_vector)
        portfolio.add_policy(policy)

    return portfolio
