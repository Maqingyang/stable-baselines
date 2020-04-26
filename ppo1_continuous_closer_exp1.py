import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines import PPO1
import os




###############################################
from particle_env_continuous_closer import PrticleEnv
env = PrticleEnv(alpha=1,beta=100,win_thre=1, max_timestep=256)
save_name = "model/ppo1_continuous_closer_exp1"
epochs = 10
timestep_per_epoch = int(1e6)
expert_n_episodes = 100
############################################



if __name__ == "__main__":




    if not os.path.exists(save_name):
        os.makedirs(save_name)
    # model = PPO2(MlpPolicy, env, verbose=0,tensorboard_log="./ppo2_particle_tensorboard/",n_cpu_tf_sess=1)
    model = PPO1(MlpPolicy,env,verbose=0,\
            timesteps_per_actorbatch=256,
            tensorboard_log=save_name,\
            policy_kwargs={"net_arch": [dict(vf=[64,64,64], pi=[64,64,64])]},\
            optim_stepsize = 3e-4,
            optim_batchsize=256,
            optim_epochs = 4,
            schedule='linear',
            n_cpu_tf_sess=16)

    for idx in range(epochs):
        model.learn(total_timesteps=timestep_per_epoch, reset_num_timesteps=False)
        model.save(save_name+"_%03dM" %((idx+1)*timestep_per_epoch/1e6))

    del model # remove to demonstrate saving and loading