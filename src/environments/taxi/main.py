import numpy as np
import json
import argparse
import os
from src.environments.taxi.Fair_Taxi_MDP_Penalty_V2 import Fair_Taxi_MDP_Penalty_V2

import tqdm
log = False
global p, fair_env
p = 1
fair_env = None


def geom_obj_score(vec, w, nsw_lambda):
    """
    Returns log(product_i (vec_i)^{w_i}) = sum_i w_i * log(vec_i)
    using the same positivity smoothing as elsewhere.
    """
    v = np.asarray(vec, dtype=float) + nsw_lambda
    v = np.where(v <= 0, nsw_lambda, v)
    return float(np.dot(w, np.log(v)))


def argmax_random_obj(R, gamma_Q, nsw_lambda, w):
    """
    Action selector for the random objective product_i u_i^{w_i}.
    R: shape (n_customers,)   (0 or accumulated R_acc)
    gamma_Q: shape (n_actions, n_customers)  (like Q[state] or gamma*Q[next])
    """
    sum_rg = R + gamma_Q  # broadcasts R onto each action row
    scores = [geom_obj_score(sum_rg[a], w, nsw_lambda) for a in range(fair_env.action_space.n)]
    if np.allclose(scores, scores[0]):
        return fair_env.action_space.sample()
    return int(np.argmax(scores))


def run_NSW_Q_learning_random(
        Q,
        do_train: bool,
        episodes: int,
        alpha: float,
        epsilon: float,
        gamma: float,
        nsw_lambda: float,
        init_val: float,
        dim_factor: float,
        tolerance: float,
        file_name: str,
        mode: str,
        non_stationary: bool,
        run: int,
        use_p_mean: bool = True, # unused
):
    global log

    if do_train:
        Q = np.zeros([fair_env.observation_space.n,
                      fair_env.action_space.n,
                      len(fair_env.loc_coords)], dtype=float) + init_val
    else:
        if Q is None:
            raise ValueError("Q must be provided when do_train=False")

    Num = np.full(fair_env.observation_space.n, epsilon, dtype=float)   # epsilon per state

    loss_data, nsw_data, total_data, p_mean_data = [], [], [], []  # p_mean_data now holds log-product scores

    best_p_mean = -np.inf          # will hold the best log-product objective
    best_R_acc = None
    best_episode = None
    full_Q_table = []

    if not do_train:
        episodes = 10
        my_range = range(1, episodes + 1)
    else:
        my_range = tqdm.tqdm(range(1, episodes + 1))

    for i in my_range:
        # Sample weights ONCE per episode (random objective fixed during the episode)
        w_full = np.random.uniform(0.5, 1.0, size=len(fair_env.loc_coords))

        R_acc = np.zeros(len(fair_env.loc_coords))
        state = fair_env.reset()
        if log:
            print(f'Episode {i}\nInitial State: {fair_env.decode(state)}')

        done = False
        old_table = np.copy(Q)
        avg_eps = []
        c = 0

        while not done:
            eps_s = Num[state]
            avg_eps.append(eps_s)

            if np.random.uniform(0, 1) < eps_s:
                action = fair_env.action_space.sample()
            else:
                if non_stationary:
                    action = argmax_random_obj(R_acc, np.power(gamma, c) * Q[state], nsw_lambda, w_full)
                else:
                    action = argmax_random_obj(0, Q[state], nsw_lambda, w_full)

            nxt, reward, done = fair_env.step(action)

            if mode == 'myopic':
                max_action = argmax_random_obj(0, gamma * Q[nxt], nsw_lambda, w_full)
            elif mode == 'immediate':
                max_action = argmax_random_obj(reward, gamma * Q[nxt], nsw_lambda, w_full)
            else:
                raise ValueError('Must have a mode')

            Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * Q[nxt, max_action] - Q[state, action]
            )

            Num[state] *= dim_factor  # diminish epsilon
            state = nxt
            R_acc += np.power(gamma, c) * reward
            c += 1

        loss = np.sum(np.abs(Q - old_table))
        loss_data.append(loss)

        R_acc_original = R_acc.copy()
        R_acc = np.where(R_acc < 0, 0, R_acc)
        R_acc = R_acc + nsw_lambda
        R_acc = np.where(R_acc <= 0, nsw_lambda, R_acc)

        nsw_score = np.power(np.prod(R_acc), 1 / len(R_acc))
        nsw_data.append(nsw_score)

        # Evaluate random objective
        R_acc_for_eval = R_acc
        w_eval = w_full

        # Use log-product (stable): sum_i w_i * log(R_i)
        rand_log_prod = geom_obj_score(R_acc_for_eval, w_eval, 0.0)
        p_mean_data.append(rand_log_prod)

        total_data.append(np.sum(R_acc))

        # Track best by the random objective (log space)
        val_p_mean = rand_log_prod
        if best_R_acc is None:
            best_R_acc = R_acc_for_eval
            best_episode = i
        if val_p_mean > best_p_mean:
            best_p_mean = val_p_mean
            best_R_acc = R_acc_for_eval
            best_episode = i

        full_Q_table.append(np.copy(Q))

        if log:
            print(f'Accumulated reward (smoothed): {R_acc}\n'
                  f'Loss: {loss}\nAverage Epsilon: {np.mean(avg_eps)}\n'
                  f'NSW (geom mean on smoothed): {nsw_score}\n'
                  f'Random Objective (log-product): {rand_log_prod}\n')

    if do_train:
        print('FINISH TRAINING NSW Q LEARNING (random objective)')
        print('Best episode: ', best_episode)
        print('Best random-objective log value: ', best_p_mean)
        Q = full_Q_table[best_episode - 1]
        print(f'Saving at policies/optimal_policy_p_{p}')
        np.save(f'policies/optimal_policy_p_{p}', Q)

    return best_p_mean, best_R_acc, p_mean_data


def run_NSW_Q_learning(Q, do_train: bool, episodes: int, alpha: float,  epsilon: float, gamma: float,
                       nsw_lambda: float, init_val: float, dim_factor: float, tolerance: float,
                       file_name: str, mode: str, non_stationary: bool, run: int, use_p_mean: bool = True):
    global log
    """
    Run welfare Q Learning

    Parameters
    ----------
    episodes : int
        number of episodes to run
    alpha : float
        learning rate
    epsilon : float
        parameter for epsilon-greedy
    gamma : float
        discount rate of the rewards
    nsw_lambda : float
        smoothing factor for calculation of nsw, which is using logs
    init_val : float
        initial value for the Q table
    dim_factor : float
        diminishing factor for epsilon
    tolerance : float
        tolerance for the online learning, if smaller than this value for 10 times, end the algorithm
        (often not used)
    file_name : str
        name of the file to store results
    mode : str
        determines whether to use myopic or immediate action selection, the final result uses "myopic" option
    non_stationary : bool
        determines the policy, whether to use stationary or non-stationary policy
    run : int
        to record the run number for stored result files
    """
    if do_train:
        Q = np.zeros([fair_env.observation_space.n, fair_env.action_space.n, len(fair_env.loc_coords)], dtype=float)
        Q = Q + init_val
    else:
        if Q is None:
            raise ValueError

    Num = np.full(fair_env.observation_space.n, epsilon, dtype=float)   # for epsilon

    loss_data, nsw_data, total_data, p_mean_data = [], [], [], []

    best_p_mean = -np.inf
    best_R_acc = None
    best_episode = None
    full_Q_table = []
    if not do_train:
        episodes = 10
        my_range = range(1, episodes+1)
    else:
        my_range = tqdm.tqdm(range(1, episodes+1))
    for i in my_range:
        R_acc = np.zeros(len(fair_env.loc_coords))
        state = fair_env.reset()
        if log:
            print('Episode {}\nInitial State: {}'.format(i,fair_env.decode(state)))
        done = False
        old_table = np.copy(Q)
        avg = []
        c = 0

        while not done:
            epsilon = Num[state]
            avg.append(epsilon)
            if np.random.uniform(0,1) < epsilon:
                action = fair_env.action_space.sample()
            else:
                if non_stationary == True:
                    action = argmax_nsw(R_acc, np.power(gamma,c)*Q[state], nsw_lambda)
                else:   # if stationary policy, then Racc doesn't affect action selection
                    action = argmax_nsw(0, Q[state], nsw_lambda)
            next, reward, done = fair_env.step(action)
            if mode == 'myopic':
                max_action = argmax_nsw(0, gamma*Q[next], nsw_lambda)
            elif mode == 'immediate':
                max_action = argmax_nsw(reward, gamma*Q[next], nsw_lambda)
            else: raise ValueError('Must have a mode')
            Q[state, action] = Q[state, action] + alpha*(reward + gamma*Q[next, max_action] - Q[state, action])


            Num[state] *= dim_factor  # epsilon diminish over time
            state = next
            R_acc += np.power(gamma,c)*reward
            c += 1

        loss = np.sum(np.abs(Q - old_table))
        loss_data.append(loss)
        if log:
            print('Racc: ', R_acc)
        R_acc_original = R_acc[::]
        R_acc = np.where(R_acc < 0, 0, R_acc) # Replace the negatives with 0
        R_acc = R_acc + nsw_lambda
        R_acc = np.where(R_acc <= 0, nsw_lambda, R_acc)
        # nsw_score = np.power(np.product(R_acc), 1/len(R_acc))
        nsw_score = np.power(np.prod(R_acc), 1/len(R_acc))
        if use_p_mean:
            p_mean = nsw(R_acc, 0) # bec the vector is already modifued, use llambda 0
        else:
            p_mean = weighted_sum(R_acc, 0)

        val_p_mean = p_mean

        if best_R_acc is None:
            best_R_acc = R_acc
            best_episode = i
        if val_p_mean>best_p_mean:
            best_p_mean = val_p_mean
            best_R_acc = R_acc
            best_episode = i
        full_Q_table.append(np.copy(Q))
        nsw_data.append(nsw_score)
        p_mean_data.append(p_mean)
        total = np.sum(R_acc)
        total_data.append(total)
        if log:
            print('Accumulated reward: {}\nLoss: {}\nAverage Epsilon: {}\nNSW: {}\n'.format(R_acc,loss,np.mean(avg),nsw_score))

    str = 'immd_' if mode == 'immediate' else ''

    if do_train:
        print('FINISH TRAINING NSW Q LEARNING')
        print('Best episode: ', best_episode)
        print('Best p mean value: ', val_p_mean)
        Q = full_Q_table[best_episode-1]
        print(f'Saving at policies/optimal_policy_p_{p}')
        np.save(f'policies/optimal_policy_p_{p}', Q)

    return best_p_mean, best_R_acc, p_mean_data


def argmax_nsw(R, gamma_Q, nsw_lambda):
    '''Helper function for run_NSW_Q_learning'''
    sum = R + gamma_Q
    nsw_vals = [nsw(sum[i], nsw_lambda) for i in range(fair_env.action_space.n)]
    if np.all(nsw_vals == nsw_vals[0]) == True: # if all values are same, random action
        # numpy argmax always return first element when all elements are same
        action = fair_env.action_space.sample()
    else:
        action = np.argmax(nsw_vals)
    return action


def nsw_old(vec, nsw_lambda):
    '''Helper function for run_NSW_Q_learning'''
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)  # replace any negative values or zeroes with lambda
    return np.sum(np.log(vec))    # numpy uses natural log


def nsw(vec, nsw_lambda):
    global p
    vec = vec + nsw_lambda
    vec = np.where(vec <= 0, nsw_lambda, vec)

    if p == 0:
        return np.prod(vec) ** (1 / len(vec))

    vec_min = np.min(vec)
    y = vec / vec_min
    z = (1/len(y)) * np.sum(np.power(y, p))
    z = np.power(z, 1/p) * vec_min

    return z


def weighted_sum(vec, nsw_lambda):
    global w
    vec = np.array(vec)*np.array(w)
    return np.sum(vec)


def get_random_policy(seed=1122, eval=False, episodes=150):
    global p, fair_env
    p = - np.inf
    file_path = f'policies/random_{seed}_policy.npy'
    print('Random Policy Call')

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the file using numpy
        print("Random policy results loaded successfully.")
        best_p_mean, best_R_acc = np.load(file_path, allow_pickle=True)
        return best_p_mean, best_R_acc

    # Default values for each argument
    fuel = 1000  # Timesteps each episode
    episodes = episodes  # Number of episodes
    alpha = 0.01  # Alpha learning rate (not used since random policy)
    alpha_N = False  # Whether to use 1/N for alpha
    epsilon = 0.10  # Always explore - this makes it random policy
    gamma = 0.999  # Discount rate
    nsw_lambda = 1e-4  # Smoothing factor
    init_val = 30  # Initial values
    dim_factor = 0.99  # Don't diminish epsilon - keep it at 1.0
    tolerance = 1e-5  # Loss threshold for Q-values between each episode
    size = 6  # Grid size of the world
    file_name = ''  # Name of .npy file
    mode = 'myopic'  # Action selection mode
    loc_coords = [[0,0], [0,5], [3,0], [1,0]]   # Location coordinates
    dest_coords = [[1,5], [5,0], [3,3], [0,3]]  # Destination coordinates
    non_stat = True  # Whether non-stationary policy

    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel,
                                        output_path='Taxi_MDP/Random_Policy/run_', fps=4)

    best_p_mean = -np.inf
    best_R_acc = None
    if not eval:
        for _ in range(3):
            p_mean, R_acc, _ = run_NSW_Q_learning_random(Q = None, do_train=True, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                                  nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                                  dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0)
            if p_mean>best_p_mean:
                best_p_mean=p_mean
                best_R_acc = R_acc
            if best_R_acc is None:
                best_R_acc = R_acc

        np.save(file_path, [best_p_mean, best_R_acc])
        return best_p_mean, best_R_acc
    else:
        print("Random policy evaluation (eval=True not applicable)")
        # For random policy, eval doesn't make sense since there's no policy to load
        # Just run the random policy
        p_mean, R_acc, _ = run_NSW_Q_learning_random(Q = None, do_train=False, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                              nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                              dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0)
        return p_mean, R_acc


def get_optimum(p_val, seed = 1122, eval=False, load_p = None, episodes=150):
    global p, fair_env
    p = p_val
    file_path = f'policies/{np.round(p_val, 3)}_{seed}_policy.npy'
    # file_path = f'policies/{p_val}_policy.npy'
    print('Oracle Call for p =', p_val)

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the file using numpy
        print("Policy loaded successfully.")
        best_p_mean, best_R_acc = np.load(file_path, allow_pickle=True)
        return best_p_mean, best_R_acc

    # Default values for each argument
    fuel = 1000  # Timesteps each episode
    episodes = episodes  # Number of episodes
    alpha = 0.01  # Alpha learning rate
    alpha_N = False  # Whether to use 1/N for alpha
    epsilon = 0.1  # Exploration rate
    gamma = 0.999  # Discount rate
    nsw_lambda = 1e-4  # Smoothing factor
    init_val = 30  # Initial values
    dim_factor = 0.9  # Diminish factor for epsilon
    tolerance = 1e-5  # Loss threshold for Q-values between each episode
    size = 6  # Grid size of the world
    file_name = ''  # Name of .npy file
    mode = 'myopic'  # Action selection mode
    loc_coords = [[0,0], [0,5], [3,0], [1,0]]   # Location coordinates
    dest_coords = [[1,5], [5,0], [3,3], [0,3]]  # Destination coordinates
    non_stat = True  # Whether non-stationary policy

    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel,
                                        output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)

    best_p_mean = -np.inf
    best_R_acc = None
    if not eval:
        for _ in range(3):
            p_mean, R_acc, _ = run_NSW_Q_learning(Q = None, do_train=True, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                                  nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                                  dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0)
            if p_mean>best_p_mean:
                best_p_mean=p_mean
                best_R_acc = R_acc
            if best_R_acc is None:
                best_R_acc = R_acc

        np.save(file_path, [best_p_mean, best_R_acc])
        return best_p_mean, best_R_acc
    else:
        print(f"Evaluating p{load_p} optimal policy at p{p_val}")
        Q = np.load(f'policies/optimal_policy_p_{load_p}.npy')
        p_mean, R_acc, _ = run_NSW_Q_learning(Q = Q, do_train=False, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                              nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                              dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0)
        return p_mean, R_acc


def get_optimum_weighted(w_val, seed = 1122, eval=False, load_p = None):
    global w, fair_env
    w = w_val
    file_path = f'policies/{"-".join([str(x) for x in w])}_{seed}_policy.npy'
    # file_path = f'policies/{p_val}_policy.npy'
    print('Oracle Call')

    # Check if the file exists
    if os.path.exists(file_path):
        # Load the file using numpy
        print("Policy loaded successfully.")
        best_p_mean, best_R_acc = np.load(file_path, allow_pickle=True)
        return best_p_mean, best_R_acc

    # Default values for each argument
    fuel = 1000  # Timesteps each episode
    episodes = 150  # Number of episodes
    alpha = 0.01  # Alpha learning rate
    alpha_N = False  # Whether to use 1/N for alpha
    epsilon = 0.1  # Exploration rate
    gamma = 0.999  # Discount rate
    nsw_lambda = 1e-4  # Smoothing factor
    init_val = 30  # Initial values
    dim_factor = 0.9  # Diminish factor for epsilon
    tolerance = 1e-5  # Loss threshold for Q-values between each episode
    size = 6  # Grid size of the world
    file_name = ''  # Name of .npy file
    mode = 'myopic'  # Action selection mode
    # loc_coords = [[0, 0], [0, 5], [3, 2]]  # Location coordinates
    # dest_coords = [[0, 4], [5, 0], [3, 3]]  # Destination coordinates
    loc_coords = [[0,0], [0,5], [3,0], [1,0]]
    dest_coords = [[1,5], [5,0], [3,3], [0,3]]
    # loc_coords = [[0,5], [4, 0], [0,3], [0,1], [4,3], [3,2]]
    # dest_coords = [[5,0], [0, 4], [3,0], [4,1], [4, 5], [3,3]]
    non_stat = True  # Whether non-stationary policy


    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel,
                                        output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)
    fair_env.seed(seed)
    # fair_env.seed(1)

    best_p_mean = -np.inf
    best_R_acc = None
    if not eval:
        for _ in range(3):
            p_mean, R_acc, _ = run_NSW_Q_learning(Q = None, do_train=True, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                                  nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                                  dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0, use_p_mean=False)
            if p_mean>best_p_mean:
                best_p_mean=p_mean
                best_R_acc = R_acc
            if best_R_acc is None:
                best_R_acc = R_acc
        np.save(file_path, [best_p_mean, best_R_acc])
        return best_p_mean, best_R_acc
    else:
        print(f"Evaluating p{load_p} optimal policy at p{p_val}")
        Q = np.load(f'policies/optimal_policy_p_{load_p}.npy')
        p_mean, R_acc, _ = run_NSW_Q_learning(Q = Q, do_train=False, episodes=episodes, alpha=alpha, epsilon=epsilon, mode=mode, gamma=gamma,
                                              nsw_lambda=nsw_lambda, init_val=init_val, non_stationary=non_stat,
                                              dim_factor=dim_factor, tolerance=tolerance, file_name=file_name, run=0)
        return p_mean, R_acc

