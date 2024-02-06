"""
Author: Minh Pham-Dinh
Created: Jan 26th, 2024
Last Modified: Feb 5th, 2024
Email: mhpham26@colby.edu

Description:
    File containing the ReplayBuffer that will be used in Dreamer.
    
    The implementation is based on:
    Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," 2019. 
    [Online]. Available: https://arxiv.org/abs/1912.01603
"""

import numpy as np
from gymnasium import Env
import torch
from addict import Dict

class ReplayBuffer:
    def __init__(self, capacity, obs_size, action_size):
        
        # check if the env is gymnasium or dmc
        self.obs_size = obs_size
        self.action_size = action_size

        # from SimpleDreamer implementation, saving memory
        state_type = np.uint8 if len(self.obs_size) < 3 else np.float32
        
        self.observation = np.zeros((capacity, ) + self.obs_size, dtype=state_type)
        
        self.actions = np.zeros((capacity, ) + self.action_size, dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pointer = 0
        self.full = False
    
        print(f'''
-----------initialized memory----------              

obs_buffer_shape: {self.observation.shape}
actions_buffer_shape: {self.actions.shape}
rewards_buffer_shape: {self.rewards.shape}
dones_buffer_shape: {self.dones.shape}

----------------------------------------
              ''')

    def add(self, obs, action, reward, done):
        """Add method for buffer

        Args:
            obs (np.array): current observation
            action (np.array): action taken
            reward (float): reward received after action
            next_obs (np.array): next observation
            done (bool): boolean value of termination or truncation
        """
        self.observation[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.observation.shape[0]
        if self.pointer == 0:
            self.full = True

    def sample(self, batch_size, seq_len, device):
        """
        Samples batches of experiences of fixed sequence length from the replay buffer, 
        taking into account the circular nature of the buffer to avoid crossing the 
        "end" of the buffer when it is full.

        This method ensures that sampled sequences are continuous and do not wrap around 
        the end of the buffer, maintaining the temporal integrity of experiences. This is 
        particularly important when the buffer is full, and the pointer marks the boundary 
        between the newest and oldest data in the buffer.

        Args:
            batch_size (int): The number of sequences to sample.
            seq_len (int): The length of each sequence to sample.
            device (torch.device): The device on which the sampled data will be loaded.

        Raises:
            Exception: If there is not enough data in the buffer to sample a full sequence.

        Returns:
            Dict: A dictionary containing the sampled sequences of observations, actions, 
            rewards, and dones. Each item in the dictionary is a tensor of shape 
            (batch_size, seq_len, feature_dimension), except for 'dones' which is of shape 
            (batch_size, seq_len, 1).

        Notes:
            - The method handles different scenarios based on the buffer's state (full or not) 
            and the pointer's position to ensure valid sequence sampling without wrapping.
            - When the buffer is not full, sequences can start from index 0 up to the 
            index where `seq_len` sequences can fit without surpassing the current pointer.
            - When the buffer is full, the method ensures sequences do not start in a way 
            that would cause them to wrap around past the pointer, effectively crossing 
            the boundary between the newest and oldest data.
            - This approach guarantees the sampled sequences respect the temporal order 
            and continuity necessary for algorithms that rely on sequences of experiences.
        """
        
        # Ensure there's enough data to sample
        if self.pointer < seq_len and not self.full:
            raise Exception('not enough data to sample')

        # handling different cases for circular sampling
        if self.full:
            if self.pointer - seq_len < 0:
                valid_range = np.arange(self.pointer, self.observation.shape[0] - (self.pointer - seq_len) + 1)
            else:
                range_1 = np.arange(0, self.pointer - seq_len + 1)
                range_2 = np.arange(self.pointer, self.observation.shape[0])
                valid_range = np.concatenate((range_1, range_2), -1)
        else:
            valid_range = np.arange(0, self.pointer-seq_len+1)

        start_index = np.random.choice(valid_range, (batch_size, 1))
        
        seq_len = np.arange(seq_len)
        sample_idcs = (start_index + seq_len) % self.observation.shape[0]
        
        batch = Dict()
        
        batch.obs = torch.from_numpy(self.observation[sample_idcs]).to(device)
        batch.actions = torch.from_numpy(self.actions[sample_idcs]).to(device)
        batch.rewards = torch.from_numpy(self.rewards[sample_idcs]).to(device)
        batch.dones = torch.from_numpy(self.dones[sample_idcs]).to(device)
        
        return batch
    
    def clear(self, ):
        self.pointer = 0
        self.full = False

    def __len__(self, ):
        return self.pointer