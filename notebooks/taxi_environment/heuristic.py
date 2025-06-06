
def budget_portfolio_with_suboptimalities(initial_p,
    K,
    get_optimum_policy,
    get_performance,
):
    """
    Build a budget-constrained portfolio up to size K.
    initial p is the smallest p you start with.
    Returns:
      portfolio   : list of (p_value, policy) for each step
    """
    portfolio = []
    for t in range(1, K+1):
        if t == 1:
            # p_t = -log2(N)
            p_t = initial_p
            policy_t = get_optimum_policy(p_t)
            portfolio.append((p_t, policy_t))
        elif t == 2:
            # p_t = 1
            p_t = 1.0
            policy_t = get_optimum_policy(p_t)
            portfolio.append((p_t, policy_t))
        else:
            # choose the interval with the smallest ratio (i.e., worst suboptimality)
            portfolio.sort(key=lambda x: x[0])
            min_ratio = 1
            chosen_interval = None

            for i in range(len(portfolio) - 1):
                p_left, policy_left = portfolio[i]
                p_right, policy_right = portfolio[i+1]

                perf_right_right = get_performance(policy_right, p_right) #optimal value for the p in the right side of the interval
                perf_left_right = get_performance(policy_left, p_right) #value of the policy in the left side of the interval evaluted w.r.t p in the right side of the interval
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

            policy_t = get_optimum_policy(p_t)
            portfolio.append((p_t, policy_t))

    portfolio.sort(key=lambda x: x[0])
    return portfolio