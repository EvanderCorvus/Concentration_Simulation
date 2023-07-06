import torch as tr
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agent_utils import NNSequential, count_parameters
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianActor(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dims, activation, act_scaling, act_positive):
        super().__init__()
        self.net = NNSequential([state_dim] + list(hidden_dims), activation)
        self.mu_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_dims[-1], act_dim)
        self.act_scaling = act_scaling
        self.act_positive = act_positive
    
    def forward(self, state, deterministic=False, with_logprob=True):
        net_out = self.net(state)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = tr.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = tr.exp(log_std)

        # Before squashing distribution
        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # Squash distribution
        pi_action = tr.tanh(pi_action)
        if self.act_positive:
            pi_action = (pi_action + 1) / 2

        pi_action = self.act_scaling * pi_action
        
        if tr.isnan(pi_action).any():
            print('piaction is nana')
        return pi_action, logp_pi
    
class QCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dims, activation):
        super().__init__()
        self.net = NNSequential([state_dim+act_dim] + list(hidden_dims) + [1], activation, output_activation=nn.LeakyReLU)
    
    def forward(self, state, action):
        q = self.net(tr.cat([state, action], dim=-1))
        return tr.squeeze(q, -1)

class SACAgent(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dims=(64,64), act_scaling=1, act_positive=False, activation=nn.LeakyReLU):
        super().__init__()
        self.actor = SquashedGaussianActor(state_dim, act_dim, hidden_dims, activation, act_scaling, act_positive)
        self.critic1 = QCritic(state_dim, act_dim, hidden_dims, activation)
        self.critic2 = QCritic(state_dim, act_dim, hidden_dims, activation)

        def act(self, state, deterministic=False):
            with tr.no_grad():
                a, _ = self.actor(state, deterministic, False)
                return a.cpu().numpy()
