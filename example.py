import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from particle_env import PrticleEnv

alpha = 10
win_thre = 0.1

def train():


    # env = gym.make('CartPole-v1')
    env = DummyVecEnv([lambda: PrticleEnv(alpha,win_thre)])
    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    # env = DummyVecEnv([lambda: env])



    model = PPO2(MlpPolicy, env, verbose=1)
    model.learn(total_timesteps=int(1e6))
    model.save("ppo2_particle")

    del model # remove to demonstrate saving and loading


def test():
    env = DummyVecEnv([lambda: PrticleEnv(alpha,win_thre)])

    model = PPO2.load("ppo2_particle")

    # Enjoy trained agent
    obs = env.reset()
    dones = False
    x = 0
    y = 0
    x_prev = 20
    y_prev = 20
    while not dones:
        if int(x)!=int(x_prev) or int(y)!=int(y_prev):
            env.render()
        x_prev, y_prev = x, y

        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        x = obs[0][0]
        y = obs[0][2]
    env.close()

if __name__ == "__main__":
    test()