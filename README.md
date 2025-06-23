# Navigating the Social Welfare Frontier: Portfolios for Multi-objective Reinforcement Learning

by Cheol Woo Kim, Jai Moondra, Shresth Verma, Madeleine Pollack, Lingkai Kong, Milind Tambe, Swati Gupta

This repository contains the code for the paper "Navigating the Social Welfare Frontier: Portfolios for Multi-objective Reinforcement Learning" presented at ICML 2025.

## Abstract

In many real-world applications of reinforcement learning (RL), deployed policies have varied impacts on different stakeholders, creating challenges in reaching consensus on how to effectively aggregate their preferences. 
Generalized p-means form a widely used class of social welfare functions for this purpose, with broad applications in fair resource allocation, AI alignment, and decision-making. 
This class includes well-known welfare functions such as Egalitarian, Nash, and Utilitarian welfare. 
However, selecting the appropriate social welfare function is challenging for decision-makers, as the structure and outcomes of optimal policies can be highly sensitive to the choice of p. 
To address this challenge, we study the concept of an α-approximate portfolio in RL, a set of policies that are approximately optimal across the family of generalized p-means for all p ∈ [−∞,1]. 
We propose algorithms to compute such portfolios and provide theoretical guarantees on the trade-offs among approximation factor, portfolio size, and computational efficiency. 
Experimental results on synthetic and real-world datasets demonstrate the effectiveness of our approach in summarizing the policy space induced by varying p values, empowering decision-makers to navigate this landscape more effectively.

## Description of the Code

This repository contains the code for the algorithms and experiments described in the paper. The code is organized into several modules, each corresponding to different components of the algorithms and experiments.

The main code to run the experiments is located in the `src` directory. The `src` directory contains the following:
- file `p_mean.py`: Contains the implementation of the p-mean functions and associated utilities.
- file `portfolio.py`: Contains the implementation of the portfolio algorithms, including `portfolio_with_line_search` (Algorithm 1 in the paper)  `budget_portfolio_with_suboptimalities` (Algorithm 3), `portfolio_of_random_policies` and `portfolio_of_random_norms` (baselines).
- directory `environments`: Contains the implementation of the environments described in the paper.

## Experiment Environments

The experiments are conducted in three environments, each designed to evaluate the performance of the portfolio algorithms across different scenarios.
The notebooks to reproduce the results in the paper are located in the `notebooks` directory.

### Resource Allocation after Natural Disaster

The basic code for this environment is located in `src/environments/natural_disaster.py`. The results in the paper can be reproduced using the Jupyter notebooks `generate_policies.ipynb` and `compute_portfolios.ipynb` in the `notebooks > natural_disaster` directory. The policies are stored in `data > natural_disaster > policy_rewards.csv`.
Note that there might be some differences in the results due to randomness in the environment. 

### Healthcare Intervention

The results in the paper can be reproduced using the Jupyter notebook `compute_portfolios.ipynb` in the `notebooks > sclm_real_world` directory. The policies are stored in `data > sclm_real_world > policy_rewards.csv`.

### Taxi Environment

The basic code for this environment is located in folder `src/environments/taxi`. We borrow the code in file `Fair_Taxi_MDP_Penalty_V2.py` from Muhang Tian. The results in the paper can be reproduced using the Jupyter notebook `compute_portfolios.ipynb` in the `notebooks > taxi` directory.

## Plots

The plots in the paper can be generated using the Jupyter notebooks in the `notebooks` directory.

### Figure 1: Portfolio for Healthcare Intervention

The Jupyter notebook `notebooks > plot_paper_figure_1.ipynb` generates Figure 1 in the paper, which shows bar charts for the portfolio of policies for the healthcare intervention environment.

### Many optimal policies for different p values

The Jupyter notebook `notebooks > many_optimal_policies.ipynb` generates a plot showing the many optimal policies for different p values for a synthetic example.
