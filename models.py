"""
Author: Minh Pham-Dinh
Created: Jan 26th, 2024
Last Modified: Jan 26th, 2024
Email: mhpham26@colby.edu

Description:
    File containing all models that will be used in Dreamer.
    
    The implementation is based on:
    Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," 2019. 
    [Online]. Available: https://arxiv.org/abs/1912.01603
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


class RSSM(nn.Module):
    """Reccurent State Space Model (RSSM)
    The main model that we will use to learn the latent dynamic of the environment
    """
    def __init__(self, stochastic_size, obs_embed_size, deterministic_size, hidden_size, action_size, activation=nn.ELU()):
        super().__init__()
        self.stochastic_size = stochastic_size
        self.action_size = action_size
        self.deterministic_size = deterministic_size
        self.obs_embed_size = obs_embed_size
        self.action_size = action_size
        
        # recurrent
        self.recurrent_linear = nn.Sequential(
            nn.Linear(stochastic_size + action_size, hidden_size),
            activation,
        )
        self.gru_cell = nn.GRUCell(hidden_size, deterministic_size)
        
        # representation model, for calculating posterior
        self.representatio_model = nn.Sequential(
            nn.Linear(deterministic_size + obs_embed_size, hidden_size),
            activation,
            nn.Linear(hidden_size, stochastic_size*2)
        )
        
        # transition model, for calculating prior, use for imagining trajectories
        self.transition_model = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            activation,
            nn.Linear(hidden_size, stochastic_size*2)
        )
        
        
    def recurrent(self, stoch_state, action, deterministic):
        """The recurrent model, calculate the deterministic state given the stochastic state
        the action, and the prior deterministic

        Args:
            a_t-1 (batch_size, action_size): action at time step, cannot be None.
            s_t-1 (batch_size, stoch_size): stochastic state at time step. Defaults to None.
            h_t-1 (batch_size, deterministic_size): deterministic at timestep. Defaults to None.

        Returns:
            h_t: deterministic at next time step
        """
        
        # initialize some sizes
        x = torch.cat((action, stoch_state), -1)
        out = self.recurrent_linear(x)
        out = self.gru_cell(out, deterministic)
        return out


    def representation(self, embed_obs, deterministic):
        """Calculate the distribution p of the stochastic state. 

        Args:
            o_t (batch_size, embeded_obs_size): embedded observation (encoded)
            h_t (batch_size, deterministic_size): determinstic size

        Returns:
            s_t posterior_distribution: distribution of stochastic states
            s_t posterior: sampled stochastic states
        """
        x = torch.cat((embed_obs, deterministic), -1)
        out = self.representatio_model(x)
        mean, std = torch.chunk(out, 2, -1)
        std = F.softplus(std) + 0.1
        
        post_dist = torch.distributions.Normal(mean, std)
        post = post_dist.rsample()
        
        return post_dist, post


    def transition(self, deterministic):
        """Calculate the distribution q of the stochastic state. 

        Args:
            h_t (batch_size, deterministic_size): determinstic size

        Returns:
            s_t prior_distribution: distribution of stochastic states
            s_t prior: sampled stochastic states
        """
        out = self.transition_model(deterministic)
        mean, std = torch.chunk(out, 2, -1)
        std = F.softplus(std) + 0.1
        
        prior_dist = torch.distributions.Normal(mean, std)
        prior = prior_dist.rsample()
        return prior_dist, prior
        

class ConvEncoder(nn.Module):
    def __init__(self, depth=32, input_shape=(3,64,64), activation=nn.ReLU()):
        super().__init__()
        self.depth = depth
        self.input_shape = input_shape
        self.conv_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=depth * 1,
                kernel_size=4,
                stride=2,
                padding="valid"
            ),
            activation,
            nn.Conv2d(
                in_channels=depth * 1,
                out_channels=depth * 2,
                kernel_size=4,
                stride=2,
                padding="valid"
            ),
            activation,
            nn.Conv2d(
                in_channels=depth * 2,
                out_channels=depth * 4,
                kernel_size=4,
                stride=2,
                padding="valid"
            ),
            activation,
            nn.Conv2d(
                in_channels=depth * 4,
                out_channels=depth * 8,
                kernel_size=4,
                stride=2,
                padding="valid"
            ),
            activation
        )
        self.conv_layer.apply(initialize_weights)
        
    def forward(self, x):
        batch_shape = x.shape[:-len(self.input_shape)]
        if not batch_shape:
            batch_shape = (1, )
        
        x = x.reshape(-1, *self.input_shape)
            
        out = self.conv_layer(x)
        
        #flatten output
        return out.reshape(*batch_shape, -1)
    

class ConvDecoder(nn.Module):
    """Decode latent dynamic
    Also referred to as observation model by the official Dreamer paper
    
    """
    def __init__(self, stochastic_size, deterministic_size, depth=32, out_shape=(3,64,64), activation=nn.ReLU()):
        super().__init__()
        self.out_shape = out_shape
        self.net = nn.Sequential(
            nn.Linear(deterministic_size + stochastic_size, depth*32),
            nn.Unflatten(1, (depth * 32, 1)),
            nn.Unflatten(2, (1, 1)),
            nn.ConvTranspose2d(
                depth * 32,
                depth * 4,
                kernel_size=5,
                stride=2,
            ),
            activation,
            nn.ConvTranspose2d(
                depth * 4,
                depth * 2,
                kernel_size=5,
                stride=2,
            ),
            activation,
            nn.ConvTranspose2d(
                depth * 2,
                depth * 1,
                kernel_size=5 + 1,
                stride=2,
            ),
            activation,
            nn.ConvTranspose2d(
                depth * 1,
                out_shape[0],
                kernel_size=5+1,
                stride=2,
            ),
        )
        self.net.apply(initialize_weights)
        
        
    def forward(self, posterior, deterministic):
        x = torch.cat((posterior, deterministic), -1)
        batch_shape = x.shape[:-1]
        if not batch_shape:
            batch_shape = (1, )
        
        x = x.reshape(-1, x.shape[-1])
        
        mean = self.net(x).reshape(*batch_shape, *self.out_shape)
        dist = torch.distributions.Normal(mean, 1)
        # #flatten output
        return torch.distributions.Independent(dist, len(self.out_shape))
    
    
class RewardNet(nn.Module):
    """reward prediction model. It take in the stochastic state and the deterministic to construct
    latent state. It then output the reward prediciton

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size, hidden_size, activation=nn.ELU()):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, stoch_state, deterministic):
        """take in the stochastic state and deterministic to construct the latent state then 
        output reard prediction

        Args:
            s_t (batch_sz, stoch_size): stochastic state (or posterior)
            h_t (batch_sz, deterministic_size): deterministic state
            
        Returns:
            r_t: rewards
        """
        x = torch.cat((stoch_state, deterministic), -1)
        return self.net(x)
    

class ContinuoNet(nn.Module):
    """continuity prediction model. It take in the stochastic state and the deterministic to construct
    latent state. It then output the prediction of whether the termination state has been reached

    Args:
        nn (_type_): _description_
    """
    def __init__(self, input_size, hidden_size, activation=nn.ELU()):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, stoch_state, deterministic):
        """take in the stochastic state and deterministic to construct the latent state then 
        output reard prediction

        Args:
            s_t stoch_state (batch_sz, stoch_size): stochastic state (or posterior)
            h_t deterministic (batch_sz, deterministic_size): deterministic state
            
        Returns:
            d_t: done or not
        """
        x = torch.cat((stoch_state, deterministic), -1)
        return self.net(x)
    
    
class Actor(nn.Module):
    """actor network
    """
    def __init__(self,
                 latent_size,
                 hidden_size,
                 action_size, 
                 discrete=True, 
                 activation=nn.ELU(), 
                 min_std=1e-4, 
                 init_std=5, 
                 mean_scale=5):
        
        super().__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.action_size = (action_size if discrete else action_size*2)
        self.discrete = discrete
        self.min_std=min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        
        self.net = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, hidden_size),
            activation,
            nn.Linear(hidden_size, self.action_size)
        )
    
        
    def forward(self, stoch_state, deterministic):
        """actor network. get in stochastic state and deterministic state to construct latent state
            and then use latent state to predict appropriate action

        Args:
            s_t stoch_state (batch_sz, stoch_size): stochastic state (or posterior)
            h_t deterministic (batch_sz, deterministic_size): deterministic state
            
        Returns:
            action distribution. OneHot if discrete, else is tanhNormal
        """
        latent_state = torch.cat((stoch_state, deterministic), -1)
        x = self.net(latent_state)
        
        if self.discrete:
            dist = torch.distributions.OneHotCategorical(logits=x)
            action = dist.sample() + dist.probs - dist.probs.detach()
        else:
            mean, std = torch.chunk(x, 2, -1)
            mean = self.mean_scale * F.tanh(mean / self.mean_scale)
            std = F.softplus(std + self.init_std) + self.min_std
            
            dist = torch.distributions.Normal(mean, std)
            dist = torch.distributions.TransformedDistribution(dist, torch.distributions.TanhTransform())
            action = torch.distributions.Independent(dist, 1).rsample()

        return action
    
    
class CriticNet(nn.Module):
    def __init__(self, ):
        super().__init__()
        