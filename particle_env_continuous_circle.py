import gym
from gym.utils import seeding
from gym import spaces, logger
import numpy as np
import math
import random


class PrticleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=10, beta=1, win_thre=1, max_timestep=-1,for_circle_traj = False):
        super(PrticleEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.mass = 1.
        self.force_mag = 10.
        self.X_lim = 10.
        self.Y_lim = 10.
        self.tau = 0.02  # seconds between state updates

        # space (x,x_dot,y_y_dot)
        high = np.array([self.X_lim,
                         np.finfo(np.float32).max,
                         self.Y_lim,
                         np.finfo(np.float32).max],
                        dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        # Example when using discrete actions:
        action_limit = np.array([self.force_mag,
                         self.force_mag],
                        dtype=np.float32)
        self.action_space = spaces.Box(low=-action_limit,high=action_limit,dtype=np.float32)
        self.x_goal = 0
        self.y_goal = 0
        self.seed()
        self.state = None
        self.alpha = alpha
        self.beta = beta
        self.win_thre = win_thre
        self.viewer = None
        self.curr_timestep = 0
        self.max_timestep = max_timestep
        self.init_dis = 0
        self.for_circle_traj = for_circle_traj
        self.shift_reward = 0
        self.velocity_reward = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, y, y_dot= state
        # action [x_force, y_force]
        force_xy = np.clip(action,-self.force_mag,self.force_mag)


 
        xy_acc = force_xy/self.mass
        x_acc = xy_acc[0]
        y_acc = xy_acc[1]

        # update state
        x = x + self.tau*x_dot
        y = y + self.tau*y_dot
        x_dot = x_dot + self.tau*x_acc
        y_dot = y_dot + self.tau*y_acc
        self.state = np.array([x,x_dot,y,y_dot])
        self.curr_timestep += 1




        done = False
        if  x < -self.X_lim \
                or x > self.X_lim \
                or y < -self.Y_lim \
                or y > self.Y_lim: 
            done = True
            # reward = -1
        distance = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
        reward = self.beta*math.exp(-self.alpha*(distance-self.win_thre))
        if distance < self.win_thre:
            reward *= 10
            done = True
        if self.max_timestep > 0 and self.curr_timestep == self.max_timestep:
            reward = -1*self.beta
            done = True

        if self.for_circle_traj:
            reward, _done = self.get_reward(x,y,x_dot,y_dot)
            done = done or _done


        return self.state, reward, done, {}

    # def get_reward(self,x,y,x_dot,y_dot):
    #     # reward for keep the same dist from the goal
    #     distance = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
    #     shift_from_init_dis = math.sqrt((distance - self.init_dis)**2)
    #     if shift_from_init_dis > 1:
    #         done = True
    #     else:
    #         done = False
    #     shift_reward = 10.*math.exp(-shift_from_init_dis)
    #     self.shift_reward = shift_reward
    #     # reward for higher speed
    #     velocity_reward = 0.1*math.sqrt(x_dot**2+y_dot**2)
    #     # velocity_reward = min(math.exp(velocity)-1,10)
    #     self.velocity_reward = velocity_reward


    #     # total_reward = shift_reward * velocity_reward
    #     total_reward = velocity_reward

    #     return total_reward, done

    def get_reward(self,x,y,x_dot,y_dot):
        # reward for keep the same dist from the goal
        distance = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
        shift_from_init_dis = math.sqrt((distance - self.init_dis)**2)
        if shift_from_init_dis > 1:
            done = True
        else:
            done = False
        shift_reward = 10.*math.exp(-shift_from_init_dis)
        self.shift_reward = shift_reward
        # reward for higher speed
        pos = np.array([x,y,0])
        vec = np.array([x_dot,y_dot,0])
        per_vec = np.cross(pos,vec)/np.linalg.norm(pos)
        velocity_reward = max(per_vec[-1], 0)
        
        # velocity_reward = min(math.exp(velocity)-1,10)
        self.velocity_reward = velocity_reward


        total_reward = velocity_reward

        return total_reward, done

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(4,dtype=np.float32)
        self.state[0] = np.random.uniform(-self.X_lim,self.X_lim)/1.5
        self.state[2] = np.random.uniform(-self.Y_lim,self.Y_lim)/1.5
        x, x_dot, y, y_dot= self.state
        self.init_dis = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
        self.curr_timestep = 0
        return self.state  


    def render(self, mode='human', close=False):
        screen_width = 600
        screen_height = 600

        world_width = self.X_lim*2
        scale = screen_width/world_width
        cartwidth = 30.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            # add goal
            goal = rendering.make_circle(20,30)
            #Add transform attribute to the cart
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goalx = screen_width/2.0 # MIDDLE OF CART
            goaly = screen_height/2.0 # MIDDLE OF CART
            #set the translation attribute of cart
            self.goaltrans.set_translation(goalx, goaly)
            self.viewer.add_geom(goal)

            # creat cart
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            # axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            cart.set_color(.8,.6,.4)

            #Add transform attribute to the cart
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            #Add Geometric cart
            self.viewer.add_geom(cart)


        if self.state is None: return None

        x,x_dot,y,y_dot = self.state
        cartx = x*scale+screen_width/2.0 # MIDDLE OF CART
        carty = y*scale+screen_height/2.0 # MIDDLE OF CART
            #set the translation attribute of cart
        self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

