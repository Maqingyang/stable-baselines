import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines import PPO1
from particle_env import PrticleEnv
import os








if __name__ == "__main__":

    # env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: PrticleEnv(alpha=1,win_thre=1, max_timestep=128)])
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])
    # env = VecNormalize(env, norm_obs=True, norm_reward=False,
    #                 clip_obs=10.)


    save_name = "ppo1_particle_infHorizon_deeperNet"
    if not os.path.exists(save_name):
        os.makedirs(save_name)
    # model = PPO2(MlpPolicy, env, verbose=0,tensorboard_log="./ppo2_particle_tensorboard/",n_cpu_tf_sess=1)
    model = PPO1(MlpPolicy,env,verbose=0,\
            tensorboard_log=save_name,\
            policy_kwargs={"net_arch": [dict(vf=[64,64,64], pi=[64,64,64])]},
            optim_batchsize=256,
            optim_epochs = 16,
            n_cpu_tf_sess=16)
    model.learn(total_timesteps=int(2e7))
    model.save(save_name+"_exp7")

    del model # remove to demonstrate saving and loading