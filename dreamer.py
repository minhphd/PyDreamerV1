"""
Author: Minh Pham-Dinh
Created: Jan 27th, 2024
Last Modified: Jan 27th, 2024
Email: mhpham26@colby.edu

Description:
    main Dreamer file.
    
    The implementation is based on:
    Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," 2019. 
    [Online]. Available: https://arxiv.org/abs/1912.01603
"""

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
        self.epsilon = self.config.main.epsilon_start
        self.env_step = 0
        
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
        self.gradient_step = 0
        
        #buffer
        self.buffer = ReplayBuffer(self.config.main.buffer_capacity, self.env, self.config.gymnasium.discrete)
        
        #tracking stuff
        self.writer = writer
    
    
    def update_epsilon(self):
        eps_start = self.config.main.epsilon_start
        eps_end = self.config.main.epsilon_end
        decay_steps = self.config.main.eps_decay_steps
        decay_rate = (eps_start - eps_end) / (decay_steps)
        self.epsilon = max(eps_end, eps_start - decay_rate*self.gradient_step)
        
        
    def train(self):
        """main training loop, implementation follow closely with the loop from the official paper

        Returns:
            _type_: _description_
        """
        
        #prefill dataset with 1 episode
        self.data_collection(self.config.main.data_init_ep)

        #main train loop
        for _ in range(self.config.main.total_iter):
            
            #training step
            for c in range(self.config.main.collect_iter):
                #draw data
                #dynamic learning
                #behavioral learning
                self.gradient_step += 1
                self.update_epsilon()
                
            # collect more data with exploration noise
            self.data_collection(self.config.main.data_interact_ep)
            
            if _ % self.config.main.eval_freq == 0:
                eval_score = self.data_collection(self.config.main.eval_eps, eval=True)
            
            
    @torch.no_grad()
    def data_collection(self, num_episodes, eval=False):
        """data collection method. Roll out agent a number of episodes and collect data
        If eval=True. The agent is set for evaluation mode with no exploration noise and data collection

        Args:
            num_episodes (int): number of episodes
            eval (bool): Evaluation mode. Defaults to False.

        Returns:
            _type_: _description_
        """
        score = 0
        ep = 0
        obs, _ = self.env.reset()
        
        #initialized all zeros
        posterior = torch.zeros((1, self.config.main.stochastic_size)).to(self.device)
        deterministic = torch.zeros((1, self.config.main.deterministic_size)).to(self.device)
        action = torch.zeros((1, self.action_size)).to(self.device)
        
        while ep < num_episodes:
            embed_obs = self.encoder(torch.from_numpy(obs).to(self.device, dtype=torch.float)) #(1, embed_obs_sz)
            deterministic = self.rssm.recurrent(action, posterior, deterministic)
            _, posterior = self.rssm.representation(embed_obs, deterministic)
            actor_out = self.actor(posterior, deterministic)
            
            # add exploration noise if not in evaluation mode
            if not eval:
                actions = actor_out.cpu().numpy()
                if self.config.gymnasium.discrete:
                    if np.random.rand() < self.epsilon:
                        action = np.random.choice(len(actions))
                    else:
                        action = np.argmax(actions)
                else:
                    mean_noise = self.config.main.mean_noise
                    std_noise = self.config.main.std_noise
                    noise = np.random.normal(mean_noise, std_noise, size=actions.shape)
                    action = (actions + noise)[0]
            else:
                actions = actor_out.cpu().numpy()
                if self.config.gymnasium.discrete:
                    action = np.argmax(actions)
                else:
                    action = actions[0]
                    
            next_obs, reward, termination, truncation, info = self.env.step(action)
            
            if not eval:
                self.buffer.add(obs, action, reward, next_obs, termination | truncation)
                self.env_step += 1
            obs = next_obs
            
            action = actor_out
            if "episode" in info:
                score += info["episode"]["r"][0]
                obs, _ = self.env.reset()
                ep += 1
            
        return score/num_episodes
    

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
    # env_id = "CarRacing-v2"
    experiment_name = config.experiment_name

    local_path = f"/{env_id}/{experiment_name}/"

    env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ResizeObservation(env, shape=(64,64))
    env = gym.wrappers.NormalizeObservation(env)
    env = channelFirst(env)
    obs, _ = env.reset()
    
    writer = SummaryWriter(config.tensorboard.log_dir + local_path)
    
    agent = Dreamer(config, env, writer)
    agent.data_collection(1)
    env.close()