"""
Author: Minh Pham-Dinh
Created: Feb 4th, 2024
Last Modified: Feb 10th, 2024
Email: mhpham26@colby.edu

Description:
    Main running file
"""


import os
os.environ["MUJOCO_GL"] = "egl"

import gymnasium as gym
import numpy as np
import yaml
from datetime import datetime
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
from utils.wrappers import *
from algos.dreamer import Dreamer
import argparse

# Define an argparse argument to accept a configuration file path from the command line
parser = argparse.ArgumentParser(description='Process configuration file path.')
parser.add_argument('--config', type=str, help='Path to the configuration file.', required=True)

# Parse the arguments
args = parser.parse_args()

# Load the configuration file specified by the command line argument
config_path = args.config
try:
    with open('configs/' + config_path, 'r') as file:
        config = Dict(yaml.load(file, Loader=yaml.FullLoader))
        print("Configuration loaded successfully.")
except FileNotFoundError:
    print(f"Configuration file {config_path} not found.")
    exit(1)
except yaml.YAMLError as e:
    print(f"Error parsing configuration file: {e}")
    exit(1)


env_id = config.env.env_id

# Prepare experiment naming and logging directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f"{config.env.env_id}_{timestamp}"
local_path = f"./{config.env.env_id}/{experiment_name}/"

wandb_writer = None
if config.wandb.enable:
    import wandb
    wandb_writer = wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        sync_tensorboard=True,
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )

if 'ALE' in config.env.env_id:
    env = gym.make(env_id, render_mode='rgb_array')
    env = AtariPreprocess(env, config.env.new_obs_size, 
                          config.video_recording.enable, 
                          record_path=config.tensorboard.log_dir + local_path + '/videos/', 
                          record_freq=config.video_recording.record_frequency)
else:
    task = config.env.task
    local_path += f"{task}/"
    env = DMCtoGymWrapper(env_id, task,
                          resize=config.env.new_obs_size,
                          record=config.video_recording.enable,
                          record_freq=config.video_recording.record_frequency,
                          record_path=config.tensorboard.log_dir + local_path + 'videos/',
                          max_episode_steps=config.env.time_limit)
    #detail: action repeat and normalize action
    env = ActionRepeat(env, config.env.action_repeat)
    env = NormalizeActions(env)
    
writer = SummaryWriter(config.tensorboard.log_dir + local_path)

# Initialize Dreamer agent and training
agent = Dreamer(config=config, env=env, writer=writer, logpath=config.tensorboard.log_dir + local_path, wandb_writer=wandb_writer)
agent.train()
env.close()
