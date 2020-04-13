import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

LR = 2e-4               # learning rate 
GAMMA = 0.98            # discount factor
EPSILON = 0.1           # clipping factor
BETA = 0.01             # entropy regularization factor
PPO_STEPS = 15          # number of PPO update steps
UPDATE_EVERY = 2048     # how many steps between network update
STD_FACTOR = 0.995      # factor to reduce std of Gaussian for action sample

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class PPOAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, network_class, state_size, action_size, fc_layer_sizes, max_t, seed):
        """Initialize an Agent object.
        
        Params
        ======
            network_class (class): class describing the network, must inherit from nn.Module
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): random seed
            double (bool): Whether or not to use Double DQN learning
            dueling (bool): Whether or not to use Dueling DQN Network architecture
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.max_t = max_t
        self.c_step = 0
        self.std_scale = 1.0

        # Network and optimizer
        self.network = network_class(state_size, action_size, fc_layer_sizes, seed).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=LR)

        # Replay memory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
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
            self.dones = []

    def act(self, state):
        """ Returns actions for given state as per current policy.
        
        Params
        ======
            state: current state
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.network.eval()
        with torch.no_grad():
            _, action_sample, log_prob, _, _ = self.network(state, scale=self.std_scale)
        self.network.train()

        # store log prob and return the clipped action sample
        self.log_probs.append(log_prob.squeeze().cpu().detach().numpy())
        return action_sample.squeeze().cpu().detach().numpy()

    def clipped_surrogate(self):
        """ Computes the clipped surrogate
        """
        # compute future rewards
        reversed_rewards = self.rewards[::-1]
        reversed_dones = self.dones[::-1]
        rewards_future_discounted = []
        c_reward = 0
        for i, reward in enumerate(reversed_rewards):
            if reversed_dones[i]:
                c_reward = 0
            c_reward = reward + GAMMA * c_reward
            rewards_future_discounted.append(c_reward)
        rewards_future_discounted = rewards_future_discounted[::-1]
        
        # normalize rewards
        rewards_norm = (rewards_future_discounted - np.mean(rewards_future_discounted)) / \
            (np.std(rewards_future_discounted) + 1e-8) 
        
        # convert to PyTorch tensors
        states = torch.tensor(self.states, dtype=torch.float, device=device)
        actions = torch.tensor(self.actions, dtype=torch.float, device=device)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float, device=device)
        rewards = torch.tensor(rewards_future_discounted, dtype=torch.float, device=device)

        # convert states to policy (or probability)
        _, _, new_log_probs, entropy, values = self.network(states, actions, self.std_scale)
        new_log_probs = new_log_probs.squeeze()
        
        # compute advantages
        advantages = rewards - values.detach()        
        
        # ratio for clipping
        # since we use log probabilities we can compute the difference and later take the exp
        ratio = (new_log_probs - old_log_probs).sum(-1).exp()

        # clipped function
        clip = torch.clamp(ratio, 1-EPSILON, 1+EPSILON)
        clipped_surrogate = torch.min(ratio*advantages, clip*advantages)

        # include a regularization term
        entropy = entropy.mean()

        return torch.mean(clipped_surrogate + BETA*entropy - F.mse_loss(values, rewards))


    def learn(self):
        """ Update value parameters using given batch of experience tuples.
        """
        for _ in range(PPO_STEPS):
            L = -self.clipped_surrogate()
            self.optimizer.zero_grad()
            L.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.)
            self.optimizer.step()
        self.std_scale *= STD_FACTOR