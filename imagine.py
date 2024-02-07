"""
Author: Minh Pham-Dinh
Created: Feb 4th, 2024
Last Modified: Feb 6th, 2024
Email: mhpham26@colby.edu

Description:
    Imagination file. Run this file to generate dream sequences
"""

import sys
import argparse
from utils.wrappers import DMCtoGymWrapper, AtariPreprocess
from addict import Dict
import yaml
import gymnasium as gym
import torch
from tqdm import tqdm
import numpy as np
import glob

parser = argparse.ArgumentParser(description='Process configuration file path.')
parser.add_argument('--runpath', type=str, help='Path to the run file.', required=True)
parser.add_argument('--horizon', type=int, help='number of imagination steps.', default=15)

# Parse the arguments
args = parser.parse_args()

# Load the configuration file specified by the command line argument
run_path = args.runpath
HORIZON = args.horizon

config_files = glob.glob(run_path + '/config/*.yml')

if len(config_files) != 1:
    print('there should only be 1 config file in config directory')

with open(config_files[0], 'r') as file:
    config = Dict(yaml.load(file, Loader=yaml.FullLoader))

env_id = config.env.env_id

if 'ALE' in config.env.env_id:
    env = gym.make(env_id, render_mode='rgb_array')
    env = AtariPreprocess(env, config.env.new_obs_size, 
                          False)
else:
    task = config.env.task
    env = DMCtoGymWrapper(env_id, task,
                          resize=config.env.new_obs_size,
                          record=False)

print("start imagining")

encode = torch.load(run_path + '/models/encoder', map_location=torch.device('cpu') )
decoder = torch.load(run_path + '/models/decoder', map_location=torch.device('cpu') )
rssm = torch.load(run_path + '/models/rssm_model', map_location=torch.device('cpu') )
actor = torch.load(run_path + '/models/actor', map_location=torch.device('cpu'))

obs, _ = env.reset()

for i in range(100):
    obs, _, _, _, _ = env.step(env.action_space.sample())

posterior = torch.zeros((1, config.main.stochastic_size))
deterministic = torch.zeros((1, config.main.deterministic_size))
e_obs = encode(torch.from_numpy(obs).to(dtype=torch.float))

_, posterior = rssm.representation(e_obs, deterministic)
    
from PIL import Image

frames = []

for i in tqdm(range(200)):
    actions = actor(posterior, deterministic)
    deterministic = rssm.recurrent(posterior, actions, deterministic)
    dist, posterior = rssm.transition(deterministic)
    d_obs = decoder(posterior, deterministic)
    d_obs = d_obs.mean.squeeze().detach().numpy()
    obs = ((d_obs.transpose([1,2,0])  + 0.5) * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(obs, "RGB")
    frames.append(img)

print("saving gif")    
frame_one = frames[0]
frame_one.save(run_path + "/imagine.gif", format="GIF", append_images=frames, save_all=True, duration=30, loop=0)
print("finished")