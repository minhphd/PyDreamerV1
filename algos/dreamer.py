"""
Author: Minh Pham-Dinh
Created: Jan 27th, 2024
Last Modified: Feb 5th, 2024
Email: mhpham26@colby.edu

Description:
    main Dreamer file.
    
    The implementation is based on:
    Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," 2019. 
    [Online]. Available: https://arxiv.org/abs/1912.01603
"""

# Standard Library Imports
import os
import numpy as np
import yaml
from tqdm import tqdm
import wandb

# Machine Learning and Data Processing Imports
import torch
import torch.nn as nn
import torch.optim as optim

# Custom Utility Imports
import utils.models as models
from utils.buffer import ReplayBuffer
from utils.utils import td_lambda, log_metrics


class Dreamer:
    def __init__(self, config, logpath, env, writer = None, wandb_writer=None):
        self.config = config
        self.device = torch.device(self.config.device)
        self.env = env
        self.obs_size = env.observation_space.shape
        self.action_size = env.action_space.n if self.config.env.discrete else env.action_space.shape[0]
        self.epsilon = self.config.main.epsilon_start
        self.env_step = 0
        self.logpath = logpath
        
        # Set random seed for reproducibility
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        
        #dynamic networks initialized
        self.rssm = models.RSSM(self.config.main.stochastic_size, 
                                self.config.main.embedded_obs_size, 
                                self.config.main.deterministic_size, 
                                self.config.main.hidden_units,
                                self.action_size).to(self.device)
        
        self.reward = models.RewardNet(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                       self.config.main.hidden_units).to(self.device)
        
        if self.config.main.continue_loss:
            self.cont_net = models.ContinuoNet(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                        self.config.main.hidden_units).to(self.device)
        
        self.encoder = models.ConvEncoder(input_shape=self.obs_size).to(self.device)
        self.decoder = models.ConvDecoder(self.config.main.stochastic_size,
                                              self.config.main.deterministic_size,
                                              out_shape=self.obs_size).to(self.device)
        self.dyna_parameters = (
            list(self.rssm.parameters())
            + list(self.reward.parameters())
            + list(self.encoder.parameters())
            + list(self.decoder.parameters())
        )

        if self.config.main.continue_loss:
          self.dyna_parameters += list(self.cont_net.parameters())
        
        #behavior networks initialized
        self.actor = models.Actor(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                  self.config.main.hidden_units,
                                  self.action_size,
                                  self.config.env.discrete).to(self.device)
        self.critic = models.Critic(self.config.main.stochastic_size + self.config.main.deterministic_size,
                                  self.config.main.hidden_units).to(self.device)
        
        #optimizers
        self.dyna_optimizer = optim.Adam(self.dyna_parameters, lr=self.config.main.dyna_model_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.main.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.main.critic_lr)
        self.gradient_step = 0
        
        #buffer
        self.buffer = ReplayBuffer(self.config.main.buffer_capacity, self.obs_size, (self.action_size, ))
        
        #tracking stuff
        self.wandb_writer = wandb_writer
        self.writer = writer
    
    
    def update_epsilon(self):
        """In use for decaying epsilon in discrete env

        Returns:
            _type_: _description_
        """
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

        #prefill dataset
        ep = 0
        obs, _ = self.env.reset()
        while ep < self.config.main.data_init_ep:
            action = self.env.action_space.sample()
            if self.config.env.discrete:  
                actions = np.zeros((self.action_size, ))
                actions[action] = 1.0
            else:
                actions = action
                
            next_obs, reward, termination, truncation, info = self.env.step(action)
        
            self.buffer.add(obs, actions, reward, termination or truncation)
            obs = next_obs
            
            if "episode" in info:
                obs, _ = self.env.reset()
                ep += 1
                print(ep)
                if 'video_path' in info and self.wandb_writer:
                    self.wandb_writer.log({'performance/videos': wandb.Video(info['video_path'], format='webm')})
                
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
        
        '''
        We unpack the batch. A batch contains:
        - b_obs (batch_size, seq_len, *obs.shape): batches of observation
        - b_a (batch_size, seq_len, 1): batches of action
        - b_r (batch_size, seq_len, 1): batches of rewards
        - b_d (batch_size, seq_len, 1): batches of termination signal
        '''
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
        posteriors = torch.zeros((batch_size, seq_len-1, self.config.main.stochastic_size)).to(self.device)
        priors = torch.zeros((batch_size, seq_len-1, self.config.main.stochastic_size)).to(self.device)
        deterministics = torch.zeros((batch_size, seq_len-1, self.config.main.deterministic_size)).to(self.device)
        
        posterior_means = torch.zeros_like(posteriors).to(self.device)
        posterior_stds = torch.zeros_like(posteriors).to(self.device)
        prior_means = torch.zeros_like(priors).to(self.device)
        prior_stds = torch.zeros_like(priors).to(self.device)

        #start passing data through the dynamic model
        for t in (range(1, seq_len)):
            deterministic = self.rssm.recurrent(posterior, b_a[:, t-1, :], deterministic)
            prior_dist, prior = self.rssm.transition(deterministic)
            
            #detail observation is shifted 1 timestep ahead(action is associated with the next state)
            posterior_dist, posterior = self.rssm.representation(eb_obs[:, t, :], deterministic)

            '''
            store recurrent data
            data are shifted 1 timestep ahead. Start from the second timestep or t=1
            '''
            posteriors[:, t-1, :] = posterior
            posterior_means[:, t-1, :] = posterior_dist.mean
            posterior_stds[:, t-1, :] = posterior_dist.scale
            
            priors[:, t-1, :] = prior
            prior_means[:, t-1, :] = prior_dist.mean
            prior_stds[:, t-1, :] = prior_dist.scale
            
            deterministics[:, t-1, :] = deterministic
            
        #we start optimizing model with the provided data
        
        '''
        Reconstruction loss. This loss helps the model learn to encode pixels observation.
        '''
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
        
        '''
        Continuity loss. This loss term helps predict the probability of an episode terminate at a particular state
        '''
        if self.config.main.continue_loss:
            # calculate log prob manually as tensorflow doesn't support float value in logprob of Bernoulli
            # follow closely to Hafner's official code for Dreamer
            cont_logits, _ = self.cont_net(posteriors, deterministics)
            cont_target = (1 - b_d[:, 1:]) * self.config.main.discount
            continue_loss = torch.nn.functional.binary_cross_entropy_with_logits(cont_logits, cont_target)
        else:
            continue_loss = torch.zeros((1)).to(self.device)
        
        '''
        KL loss. Matching the distribution of transition and representation model. This is to ensure we have the accurate transition model for use in imagination process
        '''
        priors_dist = torch.distributions.Independent(
            torch.distributions.Normal(prior_means, prior_stds), 1
        )
        posteriors_dist = torch.distributions.Independent(
            torch.distributions.Normal(posterior_means, posterior_stds), 1
        )
        kl_loss = torch.max(
            torch.mean(torch.distributions.kl.kl_divergence(posteriors_dist, priors_dist)),
            torch.tensor(self.config.main.free_nats).to(self.device)
        )
        
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
            state (batch_size, seq_len-1, stoch_state_size): starting point state
            deterministics (batch_size, seq_len-1, stoch_state_size)
        """
        
        #flatten the batches --> new size (batch_size * (seq_len-1), *)
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
        
        '''
        After imagining, we have both the state trajectories and deterministic trajectories, which can be used to create latent states.
        - state_trajectories (N, HORIZON_LEN)
        - deteerministic_trajectories (N, HORIZON_LEN)
        '''
        
        #actor update
        
        #compute rewards for each trajectories
        rewards = self.reward(state_trajectories, deterministics_trajectories)
        rewards_dist = torch.distributions.Normal(rewards, 1)
        rewards_dist = torch.distributions.Independent(rewards_dist, 1)
        rewards = rewards_dist.mode
        
        if self.config.main.continue_loss:
            _, conts_dist = self.cont_net(state_trajectories, deterministics_trajectories)
            continues = conts_dist.mean
        else:
            continues = self.config.main.discount * torch.ones_like(rewards)
        
        values = self.critic(state_trajectories, deterministics_trajectories).mode
        
        #calculate trajectories returns
        #returns should have shape (N, HORIZON_LEN - 1, 1) (last values are ignored due to nature of bootstrapping)
        returns = td_lambda(
            rewards,
            continues,
            values,
            self.config.main.lambda_,
            self.device
        )
        
        #culm product for discount
        discount = torch.cumprod(torch.cat((
            torch.ones_like(continues[:, :1]).to(self.device),
            continues[:, :-2]
        ), 1), 1).detach()
        
        # actor optimizing
        actor_loss = -(discount * returns).mean()
        
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
        
        critic_loss = -(discount.squeeze() * values_dist.log_prob(returns.detach())).mean()
        
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
            average_score: average score over number of rollout episodes
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
            deterministic = self.rssm.recurrent(posterior, action, deterministic)
            _, posterior = self.rssm.representation(embed_obs, deterministic)
            actor_out = self.actor(posterior, deterministic)
            
            #detail: add exploration noise if not in evaluation mode
            if not eval:
                actions = actor_out.cpu().numpy()
                if self.config.env.discrete:
                    if np.random.rand() < self.epsilon:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(actions)
                else:
                    mean_noise = self.config.main.mean_noise
                    std_noise = self.config.main.std_noise
                    
                    normal_dist = torch.distributions.Normal(actor_out + mean_noise, std_noise)
                    sampled_action = normal_dist.sample().cpu().numpy()
                    actions = np.clip(sampled_action, a_min=-1, a_max=1)
                    action = actions[0]
            else:
                actions = actor_out.cpu().numpy()
                if self.config.env.discrete:
                    action = np.argmax(actions)
                else:
                    actions = np.clip(actions, a_min=-1, a_max=1)
                    action = actions[0]
                    
            next_obs, reward, termination, truncation, info = self.env.step(action)
            
            if not eval:
                self.buffer.add(obs, actions, reward, termination | truncation)
                self.env_step += 1
            obs = next_obs
            
            action = actor_out
            if "episode" in info:
                cur_score = info["episode"]["r"][0]
                score += cur_score
                obs, _ = self.env.reset()
                ep += 1
                
                if 'video_path' in info and self.wandb_writer:
                    self.wandb_writer.log({'performance/videos': wandb.Video(info['video_path'], format='webm')})
                log_metrics({'performance/training score': cur_score}, self.env_step, self.writer, self.wandb_writer)
                
                posterior = torch.zeros((1, self.config.main.stochastic_size)).to(self.device)
                deterministic = torch.zeros((1, self.config.main.deterministic_size)).to(self.device)
                action = torch.zeros((1, self.action_size)).to(self.device)
            
        return score/num_episodes