Workflow description

1) Run "generate_policies.ipynb" to get reward files in "../../data/natural_disaster/" directory. This notebook calls natural_disaster.py in the "../../src/natural_disaster" directory. Before running, check all model inputs, and the output locations.

2) Edit the "__if__ name == '__main__'" block of compute_portfolio.py to match the location of the reward files generated in the previous step. Then, we can use the batch script in the slurm file to run compute_portfolios.py quickly. 

3) Run visualize_portfolio.ipynb to get the necessary plots.