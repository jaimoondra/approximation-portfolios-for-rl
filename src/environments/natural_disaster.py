from itertools import product
import random
import numpy as np
from collections import defaultdict
from scipy.sparse import csr_matrix
import random
from pprint import pprint
import csv
import pandas as pd

'''
TO DO:
08/11/2025
- Output rewards AND total allocation by cluster
- Consider how to add global constraints. For example, I want an additional reward to be given when allocating to cluster 2
'''


# Aggregate preferences into single reward
def compute_single_reward(state, action, next_state, clusters, cluster_idx, k, K, horizon, global_bonus=None):
    """
    Computes the reward for a single cluster.

    Args:
        state (tuple): Current state of unmet needs.
        action (tuple): Action taken.
        next_state (tuple): Next state of unmet needs.
        cluster_idx (int): Index of the cluster.
        k (int): Increment size for unmet need increase.

    Returns:
        float: Reward for the specified cluster.
    """
    initial_need = clusters[cluster_idx]['initial_need']
    allocation = action[cluster_idx]
    increase = max(0, next_state[cluster_idx] - state[cluster_idx]) if action[cluster_idx] == 0 else 0
    immed_rwd = max(k/10, allocation - increase)
    frac_alleviated = immed_rwd/initial_need
    # return frac_alleviated + allocation/(horizon*K)
    # return .5*allocation/(horizon*K) + .5*frac_alleviated
    base = 0.5 * allocation/(horizon*K) + 0.5 * frac_alleviated
    if global_bonus:
        boost = 0.0
        # cluster-based (IDs are 1-based in your data)
        if 'clusters' in global_bonus and (cluster_idx + 1) in global_bonus['clusters']:
            boost += global_bonus.get('weight', 0.0)
        # category-based (e.g., income == "Low-Income")
        if 'category' in global_bonus and clusters[cluster_idx].get('income') == global_bonus['category']:
            boost += global_bonus.get('weight', 0.0)
        # optional cap on total boost
        maxb = global_bonus.get('max_boost', None)
        if maxb is not None:
            boost = min(boost, maxb)
        base *= (1.0 + boost)

    return base


# Different policy types
def need_based_policy(state, action_space, cluster_data, k, K):
    max_need_idx = max(range(len(state)), key=lambda i: state[i])
    for action in action_space:
        if action[max_need_idx] == K:  # Allocate all K units to the cluster
            return action
    return random.choice(action_space)  # Fallback (shouldn't happen if action space is correct)


def per_capita_need_policy(state, action_space, cluster_data, k, K):
    per_capita_needs = [
        state[i] / cluster_data[i]["population"] if cluster_data[i]["population"] > 0 else 0
        for i in range(len(state))
    ]
    max_per_capita_idx = per_capita_needs.index(max(per_capita_needs))
    for action in action_space:
        if action[max_per_capita_idx] == K:  # Allocate all K units to the cluster
            return action
    return random.choice(action_space)  # Fallback


def population_based_policy(state, action_space, cluster_data, k, K):
    max_population_idx = max(range(len(state)), key=lambda i: cluster_data[i]["population"])
    for action in action_space:
        if action[max_population_idx] == K:  # Allocate all K units to the cluster
            return action
    return random.choice(action_space)  # Fallback


def income_based_policy(state, action_space, cluster_data, k, K):
    income_priority = {"Low-Income": 0, "Middle-Income": 1, "High-Income": 2}
    sorted_indices = sorted(
        range(len(state)),
        key=lambda i: (income_priority[cluster_data[i]["income"]], -state[i])
    )
    for idx in sorted_indices:
        for action in action_space:
            if action[idx] == K:  # Allocate all K units to the cluster
                return action
    return random.choice(action_space)  # Fallback


def proximity_based_policy(state, action_space, cluster_data, k, K):
    proximity_priority = {"Near": 0, "Far": 1}
    sorted_indices = sorted(
        range(len(state)),
        key=lambda i: (proximity_priority[cluster_data[i]["proximity"]], -state[i])
    )
    for idx in sorted_indices:
        for action in action_space:
            if action[idx] == K:  # Allocate all K units to the cluster
                return action
    return random.choice(action_space)  # Fallback


def weighted_hybrid_policy(state, action_space, cluster_data, k, K, weights):
    """
    Combines multiple rules using weighted scoring.

    Args:
        state (tuple): Current unmet needs for each cluster.
        action_space (list): List of feasible actions.
        cluster_data (list): Cluster characteristics.
        k (int): Increment size for allocations.
        K (int): Total allocation budget.
        weights (dict): Weights for different criteria.

    Returns:
        tuple: Action vector based on the weighted hybrid policy.
    """
    scores = []
    for i in range(len(state)):
        if state[i] > 0:  # Only score clusters with unmet need
            need_score = state[i] * weights.get("need", 0)
            per_capita_score = (state[i] / cluster_data[i]["population"]) * weights.get("per_capita", 0)
            income_score = weights.get("income", 0) * (1 if cluster_data[i]["income"] == "Low-Income" else 0)
            proximity_score = weights.get("proximity", 0) * (1 if cluster_data[i]["proximity"] == "Near" else 0)
            total_score = need_score + per_capita_score + income_score + proximity_score
        else:
            total_score = float('-inf')  # Ignore clusters with zero unmet need
        scores.append(total_score)

    # Allocate to the cluster with the highest score
    max_score_idx = scores.index(max(scores))
    for action in action_space:
        if action[max_score_idx] == K:
            return action
    return random.choice(action_space)


def randomized_weighted_hybrid_policy(state, action_space, cluster_data, k, K):
    """
    Randomized weighted hybrid policy with allocation in increments of k.

    Args:
        state (tuple): Current unmet needs for each cluster.
        action_space (list): List of feasible actions.
        cluster_data (list): Cluster characteristics.
        k (int): Increment size for allocations.
        K (int): Total allocation budget.

    Returns:
        tuple: Action vector based on the weighted hybrid policy.
    """
    # Randomize weights for each criterion
    weights = {
        "need": random.uniform(0, 1),
        "per_capita": random.uniform(0, 1),
        "income": random.uniform(0, 1),
        "proximity": random.uniform(0, 1),
    }
    total_weight = sum(weights.values())
    weights = {key: value / total_weight for key, value in weights.items()}  # Normalize weights

    # Calculate scores for each cluster
    scores = []
    for i in range(len(state)):
        if state[i] > 0:  # Only consider clusters with unmet need
            # print(state[i])
            need_score = state[i] * weights["need"]
            per_capita_score = (state[i] / cluster_data[i]["population"]) * weights["per_capita"]
            income_score = weights["income"] * (1 if cluster_data[i]["income"] == "Low-Income" else 0)
            proximity_score = weights["proximity"] * (1 if cluster_data[i]["proximity"] == "Near" else 0)
            scores.append(need_score + per_capita_score + income_score + proximity_score)
        else:
            scores.append(float('-inf'))  # Ignore clusters with zero unmet need

    # Allocate K in increments of k to top-scoring clusters
    action = [0] * len(state)

    # Step 1: Find the cluster with the highest score
    max_score_idx = scores.index(max(scores))

    # Step 2: Allocate K units if possible
    # print(state[max_score_idx])
    if state[max_score_idx] >= K:
        action[max_score_idx] = K
        return tuple(action)

    # Step 3: Allocate k units to the highest-scoring cluster
    action[max_score_idx] = k

    # Step 4: Distribute remaining k units randomly
    eligible_indices = [i for i in range(len(state)) if state[i] > 0 and i != max_score_idx]
    if eligible_indices:
        random_idx = random.choice(eligible_indices)
        action[random_idx] = k

    return tuple(action)


def mixed_random_policy_k_increments(state, action_space, cluster_data, k, K, deterministic_share=0.5):
    """
    Mixed random policy that allocates a share deterministically and the rest randomly in increments of k.

    Args:
        state (tuple): Current unmet needs for each cluster.
        action_space (list): List of feasible actions.
        cluster_data (list): Cluster characteristics.
        k (int): Increment size for allocations.
        K (int): Total allocation budget.
        deterministic_share (float): Proportion of K to allocate deterministically.

    Returns:
        tuple: Action vector based on the mixed random policy.
    """
    deterministic_k = round(K * deterministic_share / k) * k
    random_k = K - deterministic_k

    # Initialize action vector
    action = [0] * len(state)

    # Deterministic allocation: allocate to the highest need cluster
    max_need_idx = max(range(len(state)), key=lambda i: state[i])
    if state[max_need_idx] > 0:
        max_alloc = min(deterministic_k, (state[max_need_idx] // k) * k)
        action[max_need_idx] = max_alloc
        deterministic_k -= max_alloc

    # Random allocation: distribute remaining units across eligible clusters in increments of k
    remaining_k = random_k + deterministic_k  # Include any leftover from deterministic allocation
    eligible_indices = [i for i in range(len(state)) if state[i] > 0 and action[i] == 0]

    while remaining_k >= k and eligible_indices:
        # Randomly select a cluster with unmet need
        idx = random.choice(eligible_indices)

        # Allocate up to the minimum of k, remaining_k, or the cluster's unmet need in increments of k
        max_alloc = min((state[idx] // k) * k, k, remaining_k)
        action[idx] += max_alloc
        remaining_k -= max_alloc

        # Remove cluster from eligible list if fully satisfied
        if state[idx] - action[idx] < k:
            eligible_indices.remove(idx)

    return tuple(action)


def apply_policy(policy_name, state, action_space, cluster_data, k, K):
    """
    Applies a specified policy to generate an action.

    Args:
        policy_name (str): Name of the policy to apply.
        state (tuple): Current state of unmet needs.
        action_space (list): List of feasible actions.
        cluster_data (list): List of cluster characteristics.
        k (int): Allocation increment.
        K (int): Total allocation budget.

    Returns:
        tuple: Action vector based on the policy.
    """
    # Map policy names to functions
    policy_functions = {
        "need_based": need_based_policy,
        "per_capita": per_capita_need_policy,
        "population_based": population_based_policy,
        "income_based": income_based_policy,
        "proximity_based": proximity_based_policy,
        "weighted_hybrid": randomized_weighted_hybrid_policy,  # Add this line
    }

    policy_func = policy_functions[policy_name]
    return policy_func(state, action_space, cluster_data, k, K)


# MDP functions

def generate_feasible_next_states(current_state, action, num_clusters, k, p):
    """
    Generates all feasible next states given the current state and action.

    Args:
        current_state (tuple): Current state (unmet needs of all clusters).
        action (tuple): Action (allocation to all clusters).
        num_clusters (int): Number of clusters.
        k (int): Increment for unmet need increase.
        p (float): Probability that unmet need remains unchanged.

    Returns:
        list: List of (next_state, probability) pairs.
    """
    # Step 1: Apply action to get the interim state
    interim_state = tuple(
        max(0, current_state[i] - action[i]) for i in range(num_clusters)
    )

    # Step 2: Identify unmet clusters
    unmet_clusters = [i for i in range(num_clusters) if interim_state[i] > 0 and action[i] == 0]

    # Step 3: Generate feasible next states
    next_states = []
    if not unmet_clusters:
        # No unmet clusters, state remains unchanged
        next_states.append((interim_state, 1.0))
    else:
        # With probability p, state remains the same
        next_states.append((interim_state, p))

        # With probability (1-p), one cluster's unmet need increases
        for cluster in unmet_clusters:
            next_state = list(interim_state)
            next_state[cluster] += k
            next_states.append((tuple(next_state), (1 - p) / len(unmet_clusters)))

    return next_states


def generate_action_space(num_clusters, k, K):
    """
    Generates the full feasible action space.

    Args:
        num_clusters (int): Number of clusters.
        k (int): Allocation increment.
        K (int): Total allocation budget.

    Returns:
        list: List of feasible allocation vectors.
    """
    all_possible_actions = product(range(0, K + 1, k), repeat=num_clusters)
    feasible_actions = [
        action for action in all_possible_actions if sum(action) <= K
    ]
    return feasible_actions


def generate_valid_action(state, action_space):
    """
    Filters the action space to only include actions that allocate to clusters with unmet need > 0.

    Args:
        state (tuple): Current state of unmet needs.
        action_space (list): Precomputed feasible actions.

    Returns:
        list: Valid actions for the given state.
    """
    valid_actions = []
    for action in action_space:
        is_valid = True
        for i in range(len(state)):
            if state[i] == 0 and action[i] > 0:
                is_valid = False  # Invalid if allocating to a cluster with no unmet need
                break
        if is_valid:
            valid_actions.append(action)
    return valid_actions


def generate_state_space(initial_state, horizon, k, num_clusters):
    """
    Generates the feasible state space given the initial state and time horizon.

    Args:
        initial_state (tuple): Initial unmet needs for all clusters.
        horizon (int): Number of time steps.
        k (int): Allocation increment.
        num_clusters (int): Number of clusters.

    Returns:
        list: All feasible states over the horizon.
    """
    max_unmet_need = max(initial_state) + (horizon * k)
    feasible_states = set([initial_state])

    for t in range(horizon):
        new_states = set()
        for state in feasible_states:
            for allocation in product(range(0, k+1, k), repeat=num_clusters):
                new_state = tuple(
                    max(0, state[i] - allocation[i]) for i in range(num_clusters)
                )
                new_states.add(new_state)
        feasible_states.update(new_states)

    return sorted(feasible_states)


def generate_sparse_tpm_with_actions(states, action_space, num_clusters, k, p):
    """
    Generates a sparse TPM using precomputed action space.

    Args:
        states (list): List of all feasible states.
        action_space (list): Precomputed feasible action space.
        num_clusters (int): Number of clusters.
        k (int): Allocation increment.
        p (float): Probability that a state remains unchanged if unmet.

    Returns:
        dict: Sparse TPM, where keys are (current_state, action) pairs
              and values are lists of (next_state, probability) tuples.
    """
    state_index = {state: idx for idx, state in enumerate(states)}
    tpm = defaultdict(list)

    for current_state in states:
        for action in action_space:
            # Generate interim state
            interim_state = tuple(
                max(0, current_state[i] - action[i]) for i in range(num_clusters)
            )

            # Transition probabilities
            transitions = defaultdict(float)
            unmet_clusters = [i for i in range(num_clusters) if interim_state[i] > 0 and action[i] == 0]

            if not unmet_clusters:  # No unmet needs
                transitions[interim_state] = 1.0
            else:
                # With probability p, no unmet need increases
                transitions[interim_state] = p
                # With probability (1-p), one unmet need increases
                for cluster in unmet_clusters:
                    next_state = list(interim_state)
                    next_state[cluster] += k
                    transitions[tuple(next_state)] += (1 - p) / len(unmet_clusters)

            # Add transitions to TPM
            for next_state, prob in transitions.items():
                tpm[(current_state, action)].append((next_state, prob))

    return tpm


def generate_sparse_tpm_with_scipy(states, action_space, num_clusters, k, p):
    """
    Generates a sparse TPM using scipy's CSR matrix.

    Args:
        states (list): List of all feasible states.
        action_space (list): Precomputed feasible action space.
        num_clusters (int): Number of clusters.
        k (int): Allocation increment.
        p (float): Probability that a state remains unchanged if unmet.

    Returns:
        scipy.sparse.csr_matrix: Sparse TPM matrix of size (num_states, num_states).
        dict: Mapping of (state, action) to row indices for the TPM.
    """
    state_index = {state: idx for idx, state in enumerate(states)}
    num_states = len(states)

    # Lists to store row, column, and value data for the sparse matrix
    row_indices = []
    col_indices = []
    values = []

    for current_state in states:
        current_idx = state_index[current_state]
        for action in action_space:
            # Generate interim state
            interim_state = tuple(
                max(0, current_state[i] - action[i]) for i in range(num_clusters)
            )
            interim_idx = state_index[interim_state]

            # Transition probabilities
            transitions = defaultdict(float)
            unmet_clusters = [i for i in range(num_clusters) if interim_state[i] > 0 and action[i] == 0]

            if not unmet_clusters:  # No unmet needs
                transitions[interim_state] = 1.0
            else:
                # With probability p, no unmet need increases
                transitions[interim_state] = p
                # With probability (1-p), one unmet need increases
                for cluster in unmet_clusters:
                    next_state = list(interim_state)
                    next_state[cluster] += k
                    transitions[tuple(next_state)] += (1 - p) / len(unmet_clusters)

            # Add transitions to sparse matrix
            for next_state, prob in transitions.items():
                next_idx = state_index[next_state]
                row_indices.append(current_idx)
                col_indices.append(next_idx)
                values.append(prob)

    # Build the sparse matrix in CSR format
    tpm_sparse = csr_matrix((values, (row_indices, col_indices)), shape=(num_states, num_states))

    return tpm_sparse, state_index


def value_iteration_dynamic(states, actions, num_clusters, k, p, rewards, gamma=0.9, epsilon=1e-6):
    """
    Value iteration using dynamically generated feasible next states.

    Args:
        states (list): List of all states.
        actions (list): List of all feasible actions.
        num_clusters (int): Number of clusters.
        k (int): Increment for unmet need increase.
        p (float): Probability unmet need remains unchanged.
        rewards (dict): Reward for each state.
        gamma (float): Discount factor.
        epsilon (float): Convergence threshold.

    Returns:
        dict: Optimal value function for each state.
    """
    V = {state: 0 for state in states}
    delta = float('inf')

    while delta > epsilon:
        delta = 0
        V_new = V.copy()
        for state in states:
            max_value = float('-inf')
            for action in actions:
                next_states = generate_feasible_next_states(state, action, num_clusters, k, p)
                value = sum(prob * (rewards.get(next_state, 0) + gamma * V[next_state]) for next_state, prob in next_states)
                max_value = max(max_value, value)
            V_new[state] = max_value
            delta = max(delta, abs(V_new[state] - V[state]))
        V = V_new

    return V


def generate_random_policy(states, action_space):
    """
    Generates a random policy mapping each state to a random action.

    Args:
        states (list): List of states.
        action_space (list): List of feasible actions.

    Returns:
        dict: Policy mapping state -> action.
    """
    return {state: random.choice(action_space) for state in states}


def compute_expected_values(state, action, next_states, clusters, k, K, horizon, prob, global_bonus=None):
    """
    Computes the expected value per cluster based on the transition probabilities.

    Args:
        state (tuple): Current state.
        action (tuple): Action taken.
        next_states (list): List of (next_state, probability) pairs.
        num_clusters (int): Number of clusters.
        k (int): Allocation increment.

    Returns:
        list: Expected value added per cluster.
    """
    num_clusters = len(clusters)
    expected_values = [0] * num_clusters

    # Normalize rewards based on transition probabilities
    for next_state, prob_next in next_states:
        for i in range(num_clusters):
            # expected_values[i] += compute_single_reward(state, action, next_state, clusters, i, k, K, horizon) * prob * prob_next
            expected_values[i] += compute_single_reward(
                state, action, next_state, clusters, i, k, K, horizon, global_bonus=global_bonus
            ) * prob * prob_next

    return expected_values


def simulate_policy_dynamic_with_tpm(
        initial_state, clusters, k, K, p, horizon, action_space, policy_functions, epsilon=1e-6, global_bonus=None
):
    """
    Computes the total expected reward under dynamically selected structured policies without random sampling.
    Includes all next states with probabilities greater than epsilon.

    Args:
        initial_state (tuple): Initial state of unmet needs.
        num_clusters (int): Number of clusters.
        k (int): Increment for unmet need increase.
        p (float): Probability unmet need remains unchanged.
        horizon (int): Number of time steps.
        action_space (list): Precomputed feasible actions.
        policy_functions_list (list): List of structured policy functions to select from.
        epsilon (float): Threshold for including next states based on probability.

    Returns:
        dict: Total expected rewards for each cluster.
        dict: Final expanded policy mapping state -> action.
    """
    num_clusters = len(clusters)
    policy = {}  # Initialize an empty policy
    rewards = [0] * num_clusters  # Initialize rewards per cluster
    visited_states = set()  # Track visited states
    active_states = [(initial_state, 1.0)]  # Start with the initial state

    for t in range(horizon):
        new_active_states = []  # Track new states reachable in this time step

        for state, prob in active_states:
            # Generate valid actions for the current state
            valid_actions = generate_valid_action(state, action_space)

            if state not in policy:
                # Dynamically select a policy function for this state
                selected_policy = random.choice(list(policy_functions.values()))
                policy[state] = selected_policy(state, valid_actions, clusters, k, K)

            # Get the action from the policy
            action = policy[state]

            # Compute the feasible next states
            next_states = generate_feasible_next_states(state, action, num_clusters, k, p)

            # Compute the expected values per cluster based on next states
            expected_values_list = compute_expected_values(
                state, action, next_states, clusters, k, K, horizon, prob,global_bonus=global_bonus
            )

            # Update rewards with proper normalization
            for i in range(num_clusters):
                rewards[i] += expected_values_list[i]

            # Filter next states based on the probability threshold epsilon
            significant_next_states = [
                (next_state, prob_next) for next_state, prob_next in next_states if prob_next > epsilon
            ]

            # Add significant next states to the new active states list
            for next_state, prob_next in significant_next_states:
                if next_state not in policy:
                    # Dynamically select a policy for the next state
                    selected_policy_name, selected_policy = random.choice(list(policy_functions.items()))
                    policy[next_state] = selected_policy(next_state, valid_actions, clusters, k, K)
                if next_state not in visited_states:
                    new_active_states.append((next_state, prob * prob_next))  # Update probability for next state

            # Mark the current state as visited
            visited_states.add(state)

        # Update active states for the next time step
        active_states = new_active_states

    return rewards, policy
