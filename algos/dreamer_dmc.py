"""
Author: Minh Pham-Dinh
Created: Jan 31st, 2024
Last Modified: Jan 31st, 2024
Email: mhpham26@colby.edu

Description:
    main Dreamer file (tailored specificly for dm_control suite).
    
    The implementation is based on:
    Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," 2019. 
    [Online]. Available: https://arxiv.org/abs/1912.01603
"""

# Standard Libraries
import os
from datetime import datetime

# Third-Party Libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from addict import Dict
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2

# DeepMind Control Suite
from dm_control import suite
from dm_control.suite.wrappers import pixels

# Gymnasium
import gymnasium as gym

# Local Modules
import utils.models as models
from utils.buffer import ReplayBuffer
from utils.utils import td_lambda, get_obs, log_metrics

# Set random seed for reproducibility
SEED = 123
np.random.seed(SEED)
torch.manual_seed(SEED)

class Dreamer:
    def __init__(self, config, env, logpath, obs_size=None, writer = None, wandb_writer=None, record_ep=False, record_freq=100):
        self.record_freq = record_freq
        self.frames = []
        self.record_ep = record_ep
        self.config = config
        self.device = torch.device(self.config.device)
        self.env = env
        self.logpath=logpath
        if not obs_size:
            self.obs_size = env.observation_spec().shape
        else:
            self.obs_size = obs_size
        self.action_size = env.action_spec().shape[0]
        self.epsilon = self.config.main.epsilon_start
        self.env_step = 0
        
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
        self.cont_criterion = nn.BCELoss()
        self.dyna_optimizer = optim.Adam(self.dyna_parameters, lr=self.config.main.dyna_model_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.main.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.main.critic_lr)
        self.gradient_step = 0
        
        #buffer
        self.buffer = ReplayBuffer(self.config.main.buffer_capacity, self.obs_size, (self.action_size, ), self.config.gymnasium.discrete)
        
        #tracking stuff
        self.wandb_writer = wandb_writer
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
            
        ep = 0
        last_t = self.env.reset()
        obs = get_obs(last_t, self.obs_size[1:])
        action_spec = env.action_spec()
        while ep < 1:
        # while ep < self.config.main.data_init_ep:
            random_action = np.random.uniform(action_spec.minimum, action_spec.maximum, action_spec.shape)
                
            cur_t = self.env.step(random_action)
            next_obs = get_obs(cur_t, self.obs_size[1:])
            
            self.buffer.add(obs, 
                            random_action, 
                            cur_t.reward or 0, 
                            next_obs, cur_t.last())
            
            next_obs = obs
            
            if cur_t.last():
                last_t = self.env.reset()
                obs = get_obs(last_t, self.obs_size[1:])
                ep += 1
                print(ep)
                
        #main train loop
        for _ in tqdm(range(self.config.main.total_iter)):
            #save model if reached checkpoint
            if _ % self.config.main.save_freq == 0:
                
                #check if models folder exist
                directory = self.logpath + 'models/'
                os.makedirs(directory, exist_ok=True)
                
                #save models
                torch.save(self.rssm, self.logpath + 'models/rssm_model')
                torch.save(self.encoder, self.logpath + 'models/encoder')
                torch.save(self.decoder, self.logpath + 'models/decoder')
                torch.save(self.actor, self.logpath + 'models/actor')
                torch.save(self.critic, self.logpath + 'models/critic')
            
            #run eval if reach eval checkpoint
            if _ % self.config.main.eval_freq == 0:
                eval_score = self.data_collection(self.config.main.eval_eps, eval=True)
                metrics = {'performance/evaluation score': eval_score}
                log_metrics(metrics, self.env_step, self.writer, self.wandb_writer)
                
            #training step
            for c in tqdm(range(self.config.main.collect_iter)):
                #draw data
                batch = self.buffer.sample(self.config.main.batch_size, self.config.main.seq_len, self.device)
                
                #dynamic learning
                post, deter = self.dynamic_learning(batch)
                
                #behavioral learning
                self.behavioral_learning(post, deter)
                
                #update step
                self.gradient_step += 1
                self.update_epsilon()
                
            # collect more data with exploration noise
            self.data_collection(self.config.main.data_interact_ep)
            
    
    
    def dynamic_learning(self, batch):
        """Learning the dynamic model. In this method, we sequentially pass data in the RSSM to
        learn the model

        Args:
            batch (addict.Dict): batches of data
        """
        
        #unpack
        b_obs = batch.obs
        b_a = batch.actions
        b_r = batch.rewards
        b_d = batch.dones
        
        batch_size, seq_len, _ = b_r.shape
        eb_obs = self.encoder(b_obs)
        
        #initialized stochastic states (posterior) and deterministic states to first pass into the recurrent model
        posterior = torch.zeros((batch_size, self.config.main.stochastic_size)).to(self.device)
        deterministic = torch.zeros((batch_size, self.config.main.deterministic_size)).to(self.device)
        
        #initialized memory storing of sequential gradients data
        stochastic_size = self.config.main.stochastic_size
        
        posteriors = torch.zeros((batch_size, seq_len-1, stochastic_size)).to(self.device)
        priors = torch.zeros((batch_size, seq_len-1, stochastic_size)).to(self.device)
        deterministics = torch.zeros((batch_size, seq_len-1, stochastic_size)).to(self.device)

        kl_loss = 0

        #now the fun begin, sequentially passing data in
        #this part I got a lot of inspiration from SimpleDreamer
        for t in (range(1, seq_len)):
            deterministic = self.rssm.recurrent(posterior, b_a[:, t-1, :], deterministic)
            prior_dist, prior = self.rssm.transition(deterministic)
            posterior_dist, posterior = self.rssm.representation(eb_obs[:, t, :], deterministic)

            #store gradient data
            kl_loss += torch.distributions.kl.kl_divergence(prior_dist, posterior_dist)
            
            posteriors[:, t-1, :] = posterior
            
            priors[:, t-1, :] = prior
            
            deterministics[:, t-1, :] = deterministic
            
            
        #we start optimizing model with the provided data
        #KL loss KL(p, q)
        kl_loss = torch.max(torch.tensor(self.config.main.free_nats).to(self.device), kl_loss.mean())
        
        #reconstruction loss
        # reshape to 4 dimension because Mac Metal can only handle up to 4 dimensions
        mps_flatten = False
        if self.device == torch.device("mps"):
            mps_flatten = True
        
        reconstruct_dist = self.decoder(posteriors, deterministics, mps_flatten)
        target = b_obs[:, 1:]
        if mps_flatten:
            target = target.reshape(-1, *self.obs_size)
        
        reconstruct_loss = reconstruct_dist.log_prob(target).mean()
        
        #reward loss
        rewards = self.reward(posteriors, deterministics)
        rewards_dist = torch.distributions.Normal(rewards, 1)
        rewards_dist = torch.distributions.Independent(rewards_dist, 1)
        rewards_loss = rewards_dist.log_prob(b_r[:, 1:]).mean()
        
        #continuity loss (Bernoulli)
        continue_dist = self.cont_net(posteriors, deterministics)
        continue_loss = self.cont_criterion(continue_dist.probs, 1 - b_d[:, 1:])
        
        total_loss = self.config.main.kl_divergence_scale * kl_loss - reconstruct_loss - rewards_loss + continue_loss
        
        self.dyna_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.dyna_parameters,
            self.config.main.clip_grad,
            norm_type=self.config.main.grad_norm_type,
        )
        self.dyna_optimizer.step()
        
        #tensorboard logging
        metrics = {
            'Dynamic_model/KL': kl_loss.item(),
            'Dynamic_model/Reconstruction': reconstruct_loss.item(),
            'Dynamic_model/Reward': rewards_loss.item(),
            'Dynamic_model/Continue': continue_loss.item(),
            'Dynamic_model/Total': total_loss.item()
        }
        
        log_metrics(metrics, self.gradient_step, self.writer, self.wandb_writer)
        
        return posteriors.detach(), deterministics.detach()
    
    
    def behavioral_learning(self, state, deterministics):
        """Learning behavioral through latent imagination

        Args:
            self (_type_): _description_
            state (batch_size, seq_len, stoch_state_size): starting point state
            deterministics (batch_size, seq_len, stoch_state_size)
        """
        
        #flatten the batches
        state = state.reshape(-1, self.config.main.stochastic_size)
        deterministics = deterministics.reshape(-1, self.config.main.deterministic_size)
        
        batch_size, stochastic_size = state.shape
        _, deterministics_size = deterministics.shape
        
        #initialized trajectories
        state_trajectories = torch.zeros((batch_size, self.config.main.horizon, stochastic_size)).to(self.device)
        deterministics_trajectories = torch.zeros((batch_size, self.config.main.horizon, deterministics_size)).to(self.device)
        
        #imagine trajectories
        for t in range(self.config.main.horizon):
            # do not include the starting state
            action = self.actor(state, deterministics)
            deterministics = self.rssm.recurrent(state, action, deterministics)
            _, state = self.rssm.transition(deterministics)
            state_trajectories[:, t, :] = state
            deterministics_trajectories[:, t, :] = deterministics
        
        #now we update actor and critic
        rewards = self.reward(state_trajectories, deterministics_trajectories)
        rewards_dist = torch.distributions.Normal(rewards, 1)
        rewards_dist = torch.distributions.Independent(rewards_dist, 1)
        rewards = rewards_dist.mean
        continues = self.cont_net(state_trajectories, deterministics_trajectories).mean
        values = self.critic(state_trajectories, deterministics_trajectories).mean        
        
        #bootstrapping td_value
        returns = td_lambda(rewards, continues, values, self.config.main.lambda_, self.config.main.discount, self.device)
        
        # actor optimizing
        actor_loss = -returns.mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            self.config.main.clip_grad,
            norm_type=self.config.main.grad_norm_type,
        )
        self.actor_optimizer.step()
        
        # critic optimizing
        values_dist = self.critic(state_trajectories[:, :-1].detach(), deterministics_trajectories[:, :-1].detach())
        critic_loss = -values_dist.log_prob(returns.detach()).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            self.config.main.clip_grad,
            norm_type=self.config.main.grad_norm_type,
        )
        self.critic_optimizer.step()
        
        metrics = {
            'Behavorial_model/Actor': actor_loss.item(),
            'Behavorial_model/Critic': critic_loss.item()
        }
        
        log_metrics(metrics, self.gradient_step, self.writer, self.wandb_writer)
        
            
    @torch.no_grad()
    def data_collection(self, num_episodes, eval=False):
        """data collection method. Roll out agent a number of episodes and collect data
        If eval=True. The agent is set for evaluation mode with no exploration noise and data collection

        Args:
            num_episodes (int): number of episodes
            eval (bool): Evaluation mode. Defaults to False.
            random (bool): Random mode. Defaults to False.

        Returns:
            _type_: _description_
        """
        score = 0
        ep = 0
        
        #initialized all zeros
        posterior = torch.zeros((1, self.config.main.stochastic_size)).to(self.device)
        deterministic = torch.zeros((1, self.config.main.deterministic_size)).to(self.device)
        action_tensor = torch.zeros((1, self.action_size)).to(self.device)
        
        culm_reward = 0
        cur_t = self.env.reset()
        obs = get_obs(cur_t, self.obs_size[1:])
        
        start_rolling = False
        
        while ep < num_episodes:
            self.frames.append(cur_t.observation['pixels'])
            
            #for recording episode
            if self.record_ep and ep % self.record_freq == 0:
                if not start_rolling:
                    # make the directory if it does not exists
                    video_dir = self.logpath + 'videos/'
                    os.makedirs(video_dir, exist_ok=True)
                    height, width, _ = self.frames[0].shape
                    
                    #start recorder
                    video_writer = cv2.VideoWriter(video_dir + f'ep{ep}.webm', cv2.VideoWriter_fourcc(*'vp80'), 30, (width, height))
                    start_rolling = True
                
                video_writer.write(cv2.cvtColor(cur_t.observation['pixels'], cv2.COLOR_RGB2BGR))
            
            embed_obs = self.encoder(torch.from_numpy(obs.copy()).to(self.device, dtype=torch.float)) #(1, embed_obs_sz)
            deterministic = self.rssm.recurrent(posterior, action_tensor, deterministic)
            _, posterior = self.rssm.representation(embed_obs, deterministic)
            actor_out = self.actor(posterior, deterministic)
            actions = actor_out.cpu().numpy()
            
            # add exploration noise if not in evaluation mode
            if not eval:
                mean_noise = self.config.main.mean_noise
                std_noise = self.config.main.std_noise
                
                noise = np.random.normal(mean_noise, std_noise, size=actions.shape)
                action = (actions + noise)[0]
            else:
                action = actions[0]
                  
            cur_t = self.env.step(action)
            next_obs = get_obs(cur_t, self.obs_size[1:])
            culm_reward += (cur_t.reward or 0)
            
            if not eval:
                self.buffer.add(obs, actions, cur_t.reward or 0, next_obs, cur_t.last())
                self.env_step += 1
            obs = next_obs
            
            action_tensor = actor_out
            if cur_t.last():
                cur_score = culm_reward
                score += cur_score
                last_t = self.env.reset()
                obs = get_obs(last_t, self.obs_size[1:])
                if start_rolling:
                    video_writer.release()
                    start_rolling = False
                    if self.wandb_writer:
                        self.wandb_writer.log({'performance/videos': wandb.Video(self.logpath + f'videos/ep{ep}.webm', format='webm')})
                self.frames = []
                ep += 1
                
                log_metrics({'performance/training score': cur_score}, self.env_step, self.writer, self.wandb_writer)
                posterior = torch.zeros((1, self.config.main.stochastic_size)).to(self.device)
                deterministic = torch.zeros((1, self.config.main.deterministic_size)).to(self.device)
                action = torch.zeros((1, self.action_size)).to(self.device)
            
        return score/num_episodes
    

if __name__ == "__main__":
    # Load the configuration
    with open('./configs/dm_control/Cart-pole.yml', 'r') as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))

    domain_name = config.dmc.domain
    task = config.dmc.task

    # Generate a timestamp or use the current date
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Construct the experiment name
    experiment_name = f"{domain_name}_{task}_{timestamp}"

    # Local path for saving or accessing experiment-related files
    local_path = f"/{domain_name}/{task}/{experiment_name}/"
    
    wandb_writer = None
    if config.wandb.enable:
        import wandb
        
        wandb_writer = wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            name=experiment_name,
            save_code=True,
        )

    env = suite.load(domain_name=domain_name,
                     task_name=task)
    env = pixels.Wrapper(env, pixels_only='pixels')
    
    writer = SummaryWriter(config.tensorboard.log_dir + local_path)
    
    agent = Dreamer(config=config, obs_size=tuple([3, 64,64]), env=env, wandb_writer=wandb_writer, writer=writer, record_ep=config.video_recording.enable, logpath=config.tensorboard.log_dir + local_path)
    # agent.data_collection(1)
    agent.train()