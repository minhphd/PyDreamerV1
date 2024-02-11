"""
Author: Minh Pham-Dinh
Created: Feb 4th, 2024
Last Modified: Feb 7th, 2024
Email: mhpham26@colby.edu

Description:
    File containing wrappers for different environment types.
"""

import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np
import cv2
import os
from dm_control import suite
from dm_control.rl.control import Environment

#wrapper by Hafner et al
class ActionRepeat:
    def __init__(self, env, repeats):
        self.env = env
        self.repeats = repeats

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.repeats and not done:
            obs, reward, termination, truncation, info = self.env.step(action)
            total_reward += reward
            current_step += 1
            done = termination or truncation
        return obs, total_reward, termination, truncation, info


#wrapper by Hafner et al
class NormalizeActions:
    """
    A wrapper class that normalizes the action space of an environment.

    Args:
        env (gym.Env): The environment to be wrapped.

    Attributes:
        _env (gym.Env): The original environment.
        _mask (numpy.ndarray): A boolean mask indicating which action dimensions are finite.
        _low (numpy.ndarray): The lower bounds of the action space.
        _high (numpy.ndarray): The upper bounds of the action space.
    """

    def __init__(self, env):
        self._env = env
        self._mask = np.logical_and(
            np.isfinite(env.action_space.low),
            np.isfinite(env.action_space.high))
        self._low = np.where(self._mask, env.action_space.low, -1)
        self._high = np.where(self._mask, env.action_space.high, 1)

    def __getattr__(self, name):
        """
        Delegate attribute access to the original environment.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The value of the attribute in the original environment.
        """
        return getattr(self._env, name)

    @property
    def action_space(self):
        """
        Get the normalized action space.

        Returns:
            gym.spaces.Box: The normalized action space.
        """
        low = np.where(self._mask, -np.ones_like(self._low), self._low)
        high = np.where(self._mask, np.ones_like(self._low), self._high)
        return gym.spaces.Box(low, high, dtype=np.float32)

    def step(self, action):
        """
        Take a step in the environment with a normalized action.

        Args:
            action (numpy.ndarray): The normalized action.

        Returns:
            Tuple: A tuple containing the next state, reward, done flag, and additional information.
        """
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        original = np.where(self._mask, original, action)
        return self._env.step(original)


class DMCtoGymWrapper(gym.Env):
    """
    Wrapper to convert a DeepMind Control Suite environment to a Gymnasium environment with additional features like recording and episode truncation.

    Args:
        domain_name (str): The name of the domain.
        task_name (str): The name of the task.
        task_kwargs (dict, optional): Additional kwargs for the task.
        visualize_reward (bool, optional): Whether to visualize the reward. Defaults to False.
        resize (list, optional): New size to resize observations. Defaults to [64, 64].
        record (bool, optional): Whether to record episodes. Defaults to False.
        record_freq (int, optional): Frequency (in episodes) to record. Defaults to 100.
        record_path (str, optional): Path to save recorded videos. Defaults to '../'.
        max_episode_steps (int, optional): Maximum steps per episode for truncation. Defaults to 1000.
    """
    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, resize=[64,64], record=False, record_freq=100, record_path='../', max_episode_steps=1000, camera=None):
        super().__init__()
        self.env = suite.load(domain_name, task_name, task_kwargs=task_kwargs, visualize_reward=visualize_reward)
        self.episode_count = -1
        self.record = record
        self.record_freq = record_freq
        self.record_path = record_path
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.total_reward = 0
        self.recorder = None

        # Define action and observation space based on the DMC environment
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(low=action_spec.minimum, high=action_spec.maximum, dtype=np.float32)
        
        # Initialize the pixels wrapper for observation space
        self.env = pixels.Wrapper(self.env, pixels_only=True)
        self.resize = resize  # Assuming RGB images
        self.observation_space = gym.spaces.Box(low=-0.5, high=+0.5, shape=(3, *resize), dtype=np.float32)

        if camera is None:
            camera = dict(quadruped=2).get(domain_name, 0)
        self._camera = camera

    def step(self, action):
        time_step = self.env.step(action)
        obs = self._get_obs(self.env)
        
        reward = time_step.reward if time_step.reward is not None else 0
        self.total_reward += (reward or 0)
        self.current_step += 1
        
        termination = time_step.last()
        truncation = (self.current_step == self.max_episode_steps)
        info = {}
        if termination or truncation:
            info = {
                'episode': {
                    'r': [self.total_reward],
                    'l': self.current_step
                }
            }
            
        if self.recorder:
            frame = cv2.cvtColor(self.env.physics.render(camera_id=self._camera), cv2.COLOR_RGB2BGR)
            self.recorder.write(frame)
            video_file = os.path.join(self.record_path, f"episode_{self.episode_count}.webm")
            if termination or truncation:
                self._reset_recorder()
                info['video_path'] = video_file
        
        return obs, reward, termination, truncation, info

    def reset(self):
        self.current_step = 0
        self.total_reward = 0
        self.episode_count += 1
        
        time_step = self.env.reset()
        obs = self._get_obs(self.env)

        if self.record and self.episode_count % self.record_freq == 0:
            self._start_recording(self.env.physics.render(camera_id=self._camera))
        
        return obs, {}

    def _start_recording(self, frame):
        if not os.path.exists(self.record_path):
            os.makedirs(self.record_path)
        video_file = os.path.join(self.record_path, f"episode_{self.episode_count}.webm")
        height, width, _ = frame.shape
        self.recorder = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'vp80'), 30, (width, height))
        self.recorder.write(frame)

    def _reset_recorder(self):
        if self.recorder:
            self.recorder.release()
            self.recorder = None
            
    def _get_obs(self, env):
        obs = self.render()
        obs = obs/255 - 0.5
        rearranged_obs = obs.transpose([2,0,1])
        return rearranged_obs

    def render(self, mode='rgb_array'):
        return self.env.physics.render(*self.resize, camera_id=self._camera)  # Adjust camera_id based on the environment


class AtariPreprocess(gym.Wrapper):
    """
    A custom Gym wrapper that integrates multiple environment processing steps:
    - Records episode statistics and videos.
    - Resizes observations to a specified shape.
    - Scales and reorders observation channels.
    - Scales rewards using the tanh function.

    Parameters:
    - env (gym.Env): The original environment to wrap.
    - new_obs_size (tuple): The target size for observation resizing (height, width).
    - record (bool): If True, enable video recording.
    - record_path (str): The directory path where videos will be saved.
    - record_freq (int): Frequency (in episodes) at which to record videos.
    """
    def __init__(self, env, new_obs_size, record=False, record_path='../videos/', record_freq=100):
        super().__init__(env)
        self.env = gym.wrappers.RecordEpisodeStatistics(env)
        
        if record:
            self.env = gym.wrappers.RecordVideo(self.env, record_path, episode_trigger=lambda episode_id: episode_id % record_freq == 0)
        self.env = gym.wrappers.ResizeObservation(self.env, shape=new_obs_size)
        
        self.new_obs_size = new_obs_size
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5, 
            shape=(3, new_obs_size[0], new_obs_size[1]), 
            dtype=np.float32
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)
        obs = self.process_observation(obs)
        reward = np.tanh(reward)  # Scale reward
        return obs, reward, termination, truncation, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = self.process_observation(obs)
        return obs, info

    def process_observation(self, observation):
        """
        Process and return the observation from the environment.
        - Scales pixel values to the range [-0.5, 0.5].
        - Reorders channels to CHW format (channels, height, width).

        Parameters:
        - observation (np.ndarray): The original observation from the environment.

        Returns:
        - np.ndarray: The processed observation.
        """
        if 'pixels' in observation:
            observation = observation['pixels']
        observation = observation / 255.0 - 0.5
        observation = np.transpose(observation, (2, 0, 1))
        return observation