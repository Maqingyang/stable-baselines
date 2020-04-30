# use consecutive time steps, excluding actions as the input
# sample trajs as a semi-circle to the goal

import gym

from stable_baselines.gail import ExpertDataset, generate_expert_traj
from stable_baselines.gail.dataset.dataset import ExpertDatasetConsecutive, ExpertDatasetConsecutiveManual
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines import PPO1
import os
import matplotlib.pyplot as plt

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="9"



###############################################
from particle_env_continuous_closer_gail import PrticleEnv
from stable_baselines.gail.gail_useTrueReward_consecutiveTimeStep import GAIL
env = DummyVecEnv([lambda: PrticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=1024)])

save_name = "model/gail_exp4"
epochs = 10
timestep_per_epoch = int(1e5)
expert_n_episodes = 100
###########################################
def visualize_traj_data(traj_data):
    obs = traj_data['obs']
    obs = obs[np.random.choice(len(obs),1000)]
    x,y,x_dot,y_dot = obs[:,0],obs[:,1],obs[:,2],obs[:,3]
    for x,y,x_dot,y_dot in zip(x,y,x_dot,y_dot):
        
        plt.plot([x,x+x_dot*.1],[y,y+y_dot*.1])
    plt.show()

if __name__ == "__main__":
    


    if not os.path.exists(save_name):
        os.makedirs(save_name)

    print("\n...Generate expert trajectories\n")
    expert_obs_data = env.envs[0].sample_circle_traj() #(N,4)
    traj_data = {"obs":expert_obs_data}

    
    
    dataset = ExpertDatasetConsecutiveManual(traj_data=traj_data, traj_limitation=-1, verbose=0)
    print("...finish\n")

    visualize_traj_data(traj_data)

    model = GAIL('MlpPolicy',env, dataset, 
                 tensorboard_log=save_name, 
                 verbose=0, n_cpu_tf_sess=1)

    # Note: in practice, you need to train for 1M steps to have a working policy

    print("\n...GAIL learning\n")
    for idx in range(epochs):
        model.learn(total_timesteps=timestep_per_epoch, reset_num_timesteps=False)
        model.save(save_name+"_%03dk" %((idx+1)*timestep_per_epoch/1e3))

    # print("...finish\n")


    # del model 

