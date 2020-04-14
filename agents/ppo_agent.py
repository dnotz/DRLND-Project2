import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 2e-4               # learning rate
GAMMA = 0.99            # discount factor
EPSILON = 0.2           # clipping factor
BETA = 0.01             # entropy regularization factor
PPO_STEPS = 10          # number of PPO update steps
GAE_LAMBDA = 0.9        # lambda for generalized advantage estimate
UPDATE_EVERY = 100     # how many steps between network update
BATCH_SIZE = 256        # batch size for network updates
STD_FACTOR = 0.9995     # factor to reduce std of Gaussian for action sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, network_class, state_size, action_size, fc_layer_sizes, seed):
        """Initialize an Agent object.

        Params
        ======
            network_class (class): class describing the network, must inherit from nn.Module
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.c_step = 0
        self.std_scale = 1.0

        # Network and optimizer
        self.network = network_class(
            state_size, action_size, fc_layer_sizes, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)

        # Replay memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def step(self, state, action, reward, next_state, done):
        """ Saves step details and performs network training if enough samples have been accumulates

        Params
        ======
            state: current state
            action: action taken
            reward: reward received
            next_state: next state
            done: bool whether end of episode reached
        """
        # Save experience in memory
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

        # Learn every UPDATE_EVERY time steps.
        self.c_step = (self.c_step + 1) % UPDATE_EVERY
        if self.c_step == 0:
            # train the network
            self.learn()
            # empty memory
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.values = []
            self.dones = []

    def act(self, state):
        """ Returns actions for given state as per current policy.

        Params
        ======
            state: current state
        """
        # input state to network in eval mode and get action
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            _, action_sample, log_prob, _, value = self.network(
                state, scale=self.std_scale)
        self.network.train()

        # store log prob and value, and return the sampled action
        self.log_probs.append(list(log_prob.squeeze().cpu().detach().numpy()))
        self.values.append(list(value.squeeze().cpu().detach().numpy()))
        return action_sample.squeeze().cpu().detach().numpy()

    def learn(self):
        """ Update value parameters using given batch of experience tuples.
        """
        # remove last element since we use it as next_value to estimate the return
        self.states.pop()
        self.actions.pop()
        self.log_probs.pop()
        self.rewards.pop()
        self.dones.pop()
        c_returns = self.values.pop()

        # unpack the experiences
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        returns = []
        advantages = []
        for agent_id in range(len(self.states[0])):
            states.extend([states[agent_id] for states in self.states])
            actions.extend([actions[agent_id] for actions in self.actions])
            log_probs.extend([log_probs[agent_id]
                              for log_probs in self.log_probs])
            rewards.extend([rewards[agent_id] for rewards in self.rewards])
            dones.extend([dones[agent_id] for dones in self.dones])
            values.extend([values[agent_id] for values in self.values])

            # compute returns and advantages
            # start from last time step and use GAE
            agent_returns = []
            agent_advantages = []
            c_adv = 0
            c_return = c_returns[agent_id]
            next_value = c_returns[agent_id]
            for i in range(1, len(self.rewards)+1):
                if dones[-i]:
                    c_return = values[-i]
                    c_adv = 0
                    next_value = 0
                c_return = rewards[-i] + GAMMA * c_return
                td_error = rewards[-i] + GAMMA * next_value - values[-i]
                next_value = values[-i]
                c_adv = c_adv * GAE_LAMBDA * GAMMA + td_error
                agent_returns.append(c_return)
                agent_advantages.append(c_adv)
            # correct order and append
            returns.extend(agent_returns[::-1])
            advantages.extend(agent_advantages[::-1])

        # convert to PyTorch tensors
        states_all = torch.tensor(states, dtype=torch.float, device=device)
        actions_all = torch.tensor(actions, dtype=torch.float, device=device)
        old_log_probs_all = torch.tensor(
            log_probs, dtype=torch.float, device=device)
        rewards_all = torch.tensor(returns, dtype=torch.float, device=device)
        advantages_all = torch.tensor(
            advantages, dtype=torch.float, device=device)

        # perform PPO update steps
        for _ in range(PPO_STEPS):
            # sample batches
            batch_indices = torch.randint(
                low=0, high=states_all.shape[0], size=(BATCH_SIZE,)).long()
            states = states_all[batch_indices].detach()
            actions = actions_all[batch_indices].detach()
            old_log_probs = old_log_probs_all[batch_indices].detach()
            rewards = rewards_all[batch_indices].detach()
            advantages = advantages_all[batch_indices].detach()

            # convert states to policy (or probability)
            _, _, new_log_probs, entropy, values = self.network(
                states, actions, self.std_scale)

            # ratio for clipping
            # since we use log probabilities we can compute the difference and later take the exp
            ratio = (new_log_probs - old_log_probs).sum(-1).exp()

            # clipped function
            clip = torch.clamp(ratio, 1-EPSILON, 1+EPSILON)
            clipped_surrogate = torch.min(ratio*advantages, clip*advantages)

            # compute surrogate loss, include entropy regularization and mse loss for critic
            L = -torch.mean(clipped_surrogate + BETA *
                            entropy - 0.5 * F.mse_loss(values, rewards))

            # update weights
            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
        # reduce maximum std of Gaussian in actor network after each train step
        self.std_scale *= STD_FACTOR
