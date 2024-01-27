import gymnasium as gym
import models
from buffer import ReplayBuffer
import torch
import torch.optim as optim
import numpy as np
import yaml
from addict import Dict
from torch.utils.tensorboard import SummaryWriter

class Dreamer:
    def __init__(self, config, env: gym.Env, writer: SummaryWriter = None):
        self.config = config
        self.device = torch.device(self.config.device)
        self.env = env
        self.obs_size = env.observation_space.shape
        self.action_size = env.action_space.n if self.config.gymnasium.discrete else env.action_space.shape[0]
        
        # flip n_channels to first
        if len(self.obs_size) == 3:
            self.obs_size = (self.obs_size[-1], self.obs_size[0], self.obs_size[1])
        
        #dynamic networks initialized
        self.rssm = models.RSSM(self.config.main.stochastic_size, 
                                self.config.main.embedded_obs_size, 
                                self.config.main.deterministic_size, 
                                self.config.main.hidden_units,
                                self.action_size).to(self.device)
        
        self.reward = models.RewardNet(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                       self.config.main.hidden_units).to(self.device)
        
        # make this optional in the future
        self.cont_net = models.ContinuoNet(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                       self.config.main.hidden_units).to(self.device)
        
        self.encoder = models.ConvEncoder(input_shape=self.obs_size).to(self.device)
        self.decoder = models.ConvDecoder(self.config.main.stochastic_size,
                                              self.config.main.deterministic_size,
                                              out_shape=self.obs_size).to(self.device)
        self.dyna_parameters = (
            list(self.rssm.parameters())
            + list(self.reward.parameters())
            + list(self.cont_net.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )
        
        #behavior networks initialized
        self.actor = models.Actor(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                  self.config.main.hidden_units,
                                  self.action_size,
                                  self.config.gymnasium.discrete).to(self.device)
        self.critic = models.Critic(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                  self.config.main.hidden_units).to(self.device)
        
        #optimizers
        self.dyna_optimizer = optim.Adam(self.dyna_parameters, lr=self.config.main.dyna_model_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.main.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.main.critic_lr)
        
        #buffer
        self.buffer = ReplayBuffer(self.config.main.buffer_capacity, self.env, self.config.gymnasium.discrete)
        
        #tracking stuff
        self.writer = writer
        

if __name__ == "__main__":
    # Load the configuration
    with open('./config.yml', 'r') as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))
    
    class channelFirst(gym.ObservationWrapper):
        def __init__(self, env: gym.Env):
            gym.ObservationWrapper.__init__(self, env)
            
        def observation(self, observation):
            return observation.T

    env_id = config.gymnasium.env_id
    experiment_name = config.experiment_name

    local_path = f"/{env_id}/{experiment_name}/"

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.ResizeObservation(env, shape=(64,64))
    env = gym.wrappers.NormalizeObservation(env)
    env = channelFirst(env)
    obs, _ = env.reset()
    
    writer = SummaryWriter(config.tensorboard.log_dir + local_path)
    
    agent = Dreamer(config, env, writer)