""" 
add sample_circle_traj
fix the start point and the goal
""" 


import gym
from gym.utils import seeding
from gym import spaces, logger
import numpy as np
import math
import random


class PrticleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=10, beta=1, win_thre=1, max_timestep=-1):
        super(PrticleEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.mass = 1
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
        self.R = 4 # for circle traj
        self.x_goal = self.R
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
        self.shift_reward = 0
        self.velocity_reward = 0
        self.smallest_dist = 0
        self.trace = []
        self.prev_x,self.prev_y = 0, 0
    
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
        if abs(x-self.prev_x)+abs(y-self.prev_y) > 0.2:
            self.trace.append([x,y])



        done = False
        if  x < -self.X_lim \
                or x > self.X_lim \
                or y < -self.Y_lim \
                or y > self.Y_lim: 
            done = True
            # reward = -1

        distance = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
        reward = self.beta*max(0, self.smallest_dist-distance)/self.init_dis
        # reward = self.beta*math.exp(-self.alpha*(distance-self.win_thre))
        self.smallest_dist = min(self.smallest_dist, distance)
        
        if distance < self.win_thre:
            reward = self.beta
            done = True
        if self.max_timestep > 0 and self.curr_timestep == self.max_timestep:
            reward = -1*self.beta
            done = True



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


    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(4,dtype=np.float32)
        x, x_dot, y, y_dot= self.state
        self.init_dis = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
        self.curr_timestep = 0
        self.smallest_dist = self.init_dis
        self.trace = []
        self.prev_x,self.prev_y = x,y
        
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
        #Add trace
        if len(self.trace) > 0:
            shift_traj = [(x*scale+screen_width/2.0,y*scale+screen_height/2.0) for (x,y) in self.trace]
            # self.traj = rendering.make_polyline(shift_traj)
            # self.traj.set_linewidth(10)
            # self.viewer.add_onetime(self.traj)
            self.viewer.draw_polyline(shift_traj,linewidth=10,color=(1,0,0))
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def sample_circle_traj(self):
        sampled_circle_traj = []
        for denominator in range(1):
            denominator = 10
            tau = self.tau
            R = self.R
            force = self.force_mag
            mass = self.mass
            v_max = np.sqrt(force*R/mass)
            delta_s_max = v_max*tau
            max_delta_theta = delta_s_max/R/2 / (denominator+1)**2
            theta_dot = -max_delta_theta/tau

            sampled_theta = np.linspace(0,np.pi/2,int(np.pi/max_delta_theta))[::-1]
            x_sampled = 2*R*np.cos(sampled_theta)**2
            y_sampled = 2*R*np.cos(sampled_theta)*np.sin(sampled_theta)
            x_dot_sampled = -2*R*np.sin(2*sampled_theta)*theta_dot
            y_dot_sampled = 4*R*np.cos(2*sampled_theta)*theta_dot

            sampled_circle_traj += [(x,y,x_dot,y_dot) for x,y,x_dot,y_dot in zip(x_sampled,y_sampled,x_dot_sampled,y_dot_sampled)]
        return np.array(sampled_circle_traj)


