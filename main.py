import gymnasium as gym
import numpy as np
import yaml
from datetime import datetime
from addict import Dict
from torch.utils.tensorboard import SummaryWriter
from utils.wrappers import AtariPreprocess, DMCtoGymWrapper
from algos.dreamer import Dreamer
import os
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

env_id = config.gymnasium.env_id

# Prepare experiment naming and logging directory
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_name = f"{config.gymnasium.env_id}_{timestamp}"
local_path = f"./{config.gymnasium.env_id}/{experiment_name}/"

if config.wandb.enable:
    import wandb
    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        sync_tensorboard=True,
        name=experiment_name,
        monitor_gym=True,
        save_code=True,
    )

writer = SummaryWriter(config.tensorboard.log_dir + local_path)
    
if 'ALE' in config.gymnasium.env_id:
    env = gym.make(env_id, 1000, render_mode='rgb_array')
    env = AtariPreprocess(env, config.gymnasium.new_obs_size, 
                          config.video_recording.enable, 
                          record_path=config.tensorboard.log_dir + local_path + '/videos/', 
                          record_freq=2)

# Initialize Dreamer agent and training
agent = Dreamer(config=config, env=env, writer=writer, logpath=config.tensorboard.log_dir + local_path)
agent.train()
env.close()
