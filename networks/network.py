import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):
    """Actor (Policy) and Critic Networks"""

    def __init__(self, state_size, action_size, fc_layer_sizes, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc_layer_sizes (list of int): Layer size of each FC layer
            seed (int): Random seed
        """
        super(ActorCriticNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        self.std = nn.Parameter(torch.ones(action_size)*0.5)

        # define layers for actor network
        self.linears_act = nn.ModuleList([])
        if len(fc_layer_sizes) == 0:
            self.linears_act.append(nn.Linear(state_size, action_size))
        else:
            self.linears_act.append(nn.Linear(state_size, fc_layer_sizes[0]))
            for i in range(1, len(fc_layer_sizes)):
                self.linears_act.append(
                    nn.Linear(fc_layer_sizes[i-1], fc_layer_sizes[i]))
            self.linears_act.append(nn.Linear(fc_layer_sizes[-1], action_size))

        # define layers for critic network
        self.linears_cri = nn.ModuleList([])
        if len(fc_layer_sizes) == 0:
            self.linears_cri.append(nn.Linear(state_size, action_size))
        else:
            self.linears_cri.append(nn.Linear(state_size, fc_layer_sizes[0]))
            for i in range(1, len(fc_layer_sizes)):
                self.linears_cri.append(
                    nn.Linear(fc_layer_sizes[i-1], fc_layer_sizes[i]))
            self.linears_cri.append(nn.Linear(fc_layer_sizes[-1], 1))

    def forward(self, x, action=None, scale=1.0):
        """Build a network that maps state -> action values."""
        x_cri = x_act = x
        # actor
        for i in range(len(self.linears_act) - 1):
            x_act = F.relu(self.linears_act[i](x_act))
        # tanh maps output to [-1, 1]
        x_act = F.tanh(self.linears_act[-1](x_act))
        # compute normal, sample, log prob, entropy
        dist_norm = torch.distributions.Normal(
            x_act, scale * F.softplus(self.std))
        if action is None:
            action = dist_norm.sample()
        log_prob = dist_norm.log_prob(action)
        entropy = dist_norm.entropy().mean()

        # critic
        for i in range(len(self.linears_cri) - 1):
            x_cri = F.relu(self.linears_cri[i](x_cri))
        x_cri = self.linears_cri[-1](x_cri).squeeze()

        return x_act, action, log_prob, entropy, x_cri
