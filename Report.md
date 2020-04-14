[//]: # (Image References)

[image1]: doc/reward_plot.png "Rewards"

# Report of Project 2: Continuous Control

In the following, we will describe the PPO learning algorithm and our implementation. We also give an overview of the used hyperparameters.
Then, we show and discuss our achieved results.
Finally, we give an outlook on future work.

## Learning Algorithm

We chose to implement [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf).
PPO is an on-policy algroithm that directly optimizes the policy to maximize the expected return while ensuring in the update steps to not step too far away from the current policy. The latter is achieved by clipping applied to the objective function, which is based on the ratio of the action probabilities of the new and old policy.

For stochasticity, we train the `Actor` to predict for a state the mean and standard deviation of a normal distribution, from which actions are sampled. The `Critic` estimates the value of a state. These value estimates are used together with the agents' returns to compute advantages. We compute [generalized advantage estimates (GAE)](https://arxiv.org/pdf/1506.02438.pdf) as they reduce the variance while keeping the bias small.

The overall loss used for training is a combination of PPO's clipped surrogate loss, an entropy regularization term to encourage exploration and the mean squared loss between the critic's value estimates and the returns.
Since we use the second version of the environment we have multiple (non-interacting, parallel) copies of the same agent to collect experiences. We found this method of using de-correlated experiences particularly beneficial.

## Hyperparameters

We train our PPO agent for up to `n_episodes = 5000` episodes (far fewer were needed), with up to `max_t = 1000` timesteps each.

For the agent training we use the following hyperparameters:

* **`LR = 2e-4`**
* **`GAMMA = 0.99`**
* **`EPSILON = 0.2`**
* **`BETA = 0.01`**
* **`PPO_STEPS = 10`**
* **`GAE_LAMBDA = 0.9`**
* **`UPDATE_EVERY = 100`**
* **`BATCH_SIZE = 256`**
* **`STD_FACTOR = 0.9995`**

`LR` is the learning rate of the `Adam` optimizer for updating the network weights. These weights are updated every `UPDATE_EVERY` timesteps and each update consists of `PPO_STEPS` update steps. For each update step we randomly sample a batch size of `BATCH_SIZE` experiences.
Rewards are discounted with `GAMMA` and `GAE_LAMBDA` refers to the lambda value for the computation of the generalized advantage estimates. `EPSILON` is the clipping factor applied to the ratio of new and old probabilities. Finally, we continuously decrease the standard deviation factor of the normal distribution (starting from 1) with `STD_FACTOR` after every update. `BETA` is the factor of the entropy regularization term added to the loss.

We use fully-connected layers for our network architecture with the following specifications:

* Actor: **`LAYER_SIZES = [256, 128, 4]`**
* Critic: **`LAYER_SIZES = [256, 128, 1]`**

The first two linear layers of the `Actor` are followed by `ReLU` activations, whereas the last one is followed by a `tanh` activation to ensure an output in [-1,1].
The first two linear layers of the `Critic` are followed by `ReLU` activations.

## Achieved Scores (Rewards)

We have trained the described PPO agent on the second version of the environment with 20 parallel agent copies. After just 64 episodes (with 1000 steps each) the environment was solved, i.e. the average score reached 30 over 100 consecutive episodes and over all agents.
Below you can see a plot of the scores (rewards) against the number of episodes trained.

![Rewards][image1]

## Ideas for Future Work

While the trained agent performs very well there are many directions for future work.
Instead of having two completely separate neworks for actor and critic they could both share a common base network.
Furthermore, it would be great to make statistically significant claims about the agent's performance. To do so, we could repeat the same experiment several times and compute the mean and standard deviation of the agent's performance.
An interesting direction for research is to make PPO work on the first version of the environment, which only has a single agent. In our experiments, due to the strong correlation of the samples, reaching an average score of 30 seemed challenging. We managed to received an average score of 18 but noticed strong fluctuations in performance.
Last but not least, it would be interesting to compare the PPO performance to implementations of other promising algorithms, such as [A3C](https://arxiv.org/pdf/1602.01783.pdf) or [D4PG](https://openreview.net/pdf?id=SyZipzbCb).
