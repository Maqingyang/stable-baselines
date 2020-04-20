import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2
from stable_baselines import PPO1
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"



from particle_env_continuous_circle_maxReward import PrticleEnv
save_name = "model/part_circle_exp2"
epochs = 20

if __name__ == "__main__":

    env = DummyVecEnv([lambda: PrticleEnv(alpha=1,beta=10,win_thre=1, max_timestep=256,for_circle_traj=True)])



    if not os.path.exists(save_name):
        os.makedirs(save_name)
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
        model.learn(total_timesteps=int(1e5), reset_num_timesteps=False)
        model.save(save_name+"_epoch%02d" %(idx+1))

    del model # remove to demonstrate saving and loading