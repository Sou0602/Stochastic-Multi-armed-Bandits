# Stochastic-Multi-armed-Bandits

This is an implementation of the following regret minimisation algorithms of Stochastic Multi-armed-Bandits with rewards following a Bernoulli Distribution:
1. Epsilon Greedy
2. UCB
3. KL-UCB
4. Thompson Sampling

The bandit.py file in the submissions folder has the functions to all the algorithms. The instances folder has the information of the true means and the number if bandits for an instance.

While calling the bandit.py file use the following syntax in the input:
bandit.py "instance" "algorithm" "randomSeed" "epsilon" "horizon"

Will update the plots for the algorithm when ready.
