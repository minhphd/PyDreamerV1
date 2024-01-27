import numpy as np
from gymnasium import Env
import torch

class ReplayBuffer:
    def __init__(self, capacity, env: Env, continuous: bool):
        self.obs_size = env.observation_space.shape
        self.action_size = env.action_space.shape if continuous else (1,)
        
        # flip n_channels to first
        if len(self.obs_size) == 3:
            self.obs_size = (self.obs_size[-1], self.obs_size[0], self.obs_size[1])
        
        # from SimpleDreamer implementation, saving memory
        state_type = np.uint8 if len(self.obs_size) < 3 else np.float32
        
        self.observation = np.zeros((capacity, ) + self.obs_size, dtype=state_type)
        self.actions = np.zeros((capacity, ) + self.action_size, dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observation = np.zeros((capacity, ) + self.obs_size, dtype=state_type)
        if not continuous:
            self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.pointer = 0
        self.full = False
    
        print(f'''
-----------initialized memory----------              

obs_buffer_shape: {self.observation.shape}
actions_buffer_shape: {self.actions.shape}
rewards_buffer_shape: {self.rewards.shape}
nxt_obs_buffer_shape: {self.next_observation.shape}
dones_buffer_shape: {self.dones.shape}

----------------------------------------
              ''')

    def add(self, obs, action, reward, next_obs, done):
        self.observation[self.pointer] = obs
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_observation[self.pointer] = next_obs
        self.dones[self.pointer] = done
        self.pointer = (self.pointer + 1) % self.observation.shape[0]
        if self.pointer == 0:
            self.full = True

    def sample(self, batch_size, seq_len, device):
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
        
        obs = torch.from_numpy(self.observation[sample_idcs]).to(device)
        actions = torch.from_numpy(self.actions[sample_idcs]).to(device)
        rewards = torch.from_numpy(self.rewards[sample_idcs]).to(device)
        next_obs = torch.from_numpy(self.next_observation[sample_idcs]).to(device)
        dones = torch.from_numpy(self.dones[sample_idcs]).to(device)
        
        return obs, actions, rewards, next_obs, dones
    
    def clear(self, ):
        self.pointer = 0

    def __len__(self, ):
        return self.pointer