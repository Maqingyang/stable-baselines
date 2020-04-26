## First, sanity test on GAIL by training a model to simulate the circle traj. 

import gym

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



###############################################
from particle_env_continuous_closer import PrticleEnv
from stable_baselines.gail.gail_useTrueReward import GAIL
env = DummyVecEnv([lambda: PrticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=256)])
save_name = "model/gail_exp2"
epochs = 10
timestep_per_epoch = int(1e6)
expert_n_episodes = 100
############################################

if __name__ == "__main__":

    if not os.path.exists(save_name):
        os.makedirs(save_name)

    # Generate expert trajectories (train expert)
    print("\n...Generate expert trajectories\n")
    model = PPO1.load("model/part_circle_exp2_epoch05_sib.zip")
    from particle_env_continuous_circle_maxReward import PrticleEnv as Expert_ParticleEnv

    env_expert = Expert_ParticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=256,for_circle_traj=True)
    model.set_env(env_expert)
    generate_expert_traj(model, 'expert_part_circle_exp2_epoch05_sib', n_episodes=expert_n_episodes)
    print("...finish\n")


    # Load the expert dataset
    print("\n...Load the expert dataset\n")

    dataset = ExpertDataset(expert_path='expert_part_circle_exp2_epoch05_sib.npz', traj_limitation=-1, verbose=1)
    print("...finish\n")

    model = GAIL('MlpPolicy',env, dataset, 
                 tensorboard_log=save_name, verbose=0, n_cpu_tf_sess=None)

    # Note: in practice, you need to train for 1M steps to have a working policy

    print("\n...GAIL learning\n")
    for idx in range(epochs):
        model.learn(total_timesteps=timestep_per_epoch, reset_num_timesteps=False)
        model.save(save_name+"_%03dM" %((idx+1)*timestep_per_epoch/1e6))

    print("...finish\n")


    del model 

