## First, sanity test on GAIL by training a model to simulate the circle traj. 

import gym

from stable_baselines import GAIL, SAC
from stable_baselines.gail import ExpertDataset, generate_expert_traj

import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import PPO1
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

# from particle_env_continuous import PrticleEnv
from particle_env_continuous_circle_maxReward import PrticleEnv


# Generate expert trajectories (train expert)
env = PrticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=256,for_circle_traj=True)

model = PPO1.load("model/part_circle_exp2_epoch05_sib.zip")
model.set_env(env)
generate_expert_traj(model, 'expert_part_circle_exp2_epoch05_sib', n_episodes=10)


# Load the expert dataset
dataset = ExpertDataset(expert_path='expert_part_circle_exp2_epoch05_sib.npz', traj_limitation=10, verbose=1)

model = GAIL('MlpPolicy'\
             ,DummyVecEnv([lambda: PrticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=256,for_circle_traj=True)])\
             , dataset, verbose=1, n_cpu_tf_sess=None)

# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=int(1e4))
model.save("_gail_sanity_test_exp1")


del model 

# %%
