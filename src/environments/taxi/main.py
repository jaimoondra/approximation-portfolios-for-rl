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
        print('Best val p: ', val_p_mean)
        Q = full_Q_table[best_episode-1]
        # print(f'Saving at policies/optimal_policy_p_{p}')
        # np.save(f'policies/optimal_policy_p_{p}', Q)

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
    sum = np.sum(np.power(vec, p))
    return np.power(sum / len(vec), 1/p)


def weighted_sum(vec, nsw_lambda):
    global w
    vec = np.array(vec)*np.array(w)
    return np.sum(vec)


def get_optimum(p_val, seed = 1122, eval=False, load_p = None):
    global p, fair_env
    p = p_val
    file_path = f'policies/{np.round(p_val, 3)}_{seed}_policy.npy'
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
    loc_coords = [[0,0], [0,5], [3,0], [1,0]]   # Location coordinates
    dest_coords = [[1,5], [5,0], [3,3], [0,3]]  # Destination coordinates
    non_stat = True  # Whether non-stationary policy


    fair_env = Fair_Taxi_MDP_Penalty_V2(size, loc_coords, dest_coords, fuel,
                                        output_path='Taxi_MDP/NSW_Q_learning/run_', fps=4)
    # fair_env.seed(seed)

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

        # np.save(file_path, [best_p_mean, best_R_acc])
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

