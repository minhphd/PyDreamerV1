import gymnasium as gym
from dm_control import suite
from dm_control.suite.wrappers import pixels
import numpy as np
import cv2
import os
from utils.utils import get_obs

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
    def __init__(self, domain_name, task_name, task_kwargs=None, visualize_reward=False, resize=[3, 64,64], record=False, record_freq=100, record_path='../', max_episode_steps=1000):
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
        obs_shape = tuple(resize)  # Assuming RGB images
        self.observation_space = gym.spaces.Box(low=-0.5, high=+0.5, shape=obs_shape, dtype=np.float32)

    def step(self, action):
        time_step = self.env.step(action)
        obs = get_obs(self.env, self.observation_space.shape[1:])
        
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
            frame = cv2.cvtColor(time_step.observation['pixels'], cv2.COLOR_RGB2BGR)
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
        obs = get_obs(self.env, self.observation_space.shape[1:])

        if self.record and self.episode_count % self.record_freq == 0:
            self._start_recording(time_step.observation['pixels'])
        
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

    def render(self, mode='rgb_array'):
        return self.env.physics.render(camera_id=0)  # Adjust camera_id based on the environment
