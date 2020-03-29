import gym
from gym.utils import seeding
from gym import spaces, logger
import numpy as np
import math
import random


class PrticleEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, alpha=10, win_thre=1, max_timestep=-1):
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
        self.action_space = spaces.Discrete(4)
        self.x_goal = 0
        self.y_goal = 0
        self.seed()
        self.state = None
        self.alpha = alpha
        self.win_thre = win_thre
        self.viewer = None
        self.curr_timestep = 0
        self.max_timestep = max_timestep
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        state = self.state
        x, x_dot, y, y_dot= state
        # 0 up
        # 1 down
        # 2 left
        # 3 right

        if   action==0: force = np.array([0, self.force_mag])
        elif action==1: force = np.array([0, -self.force_mag]) 
        elif action==2: force = np.array([-self.force_mag, 0]) 
        elif action==3: force = np.array([self.force_mag, 0]) 
        xy_acc = force/self.mass
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
        reward = math.exp(-self.alpha*(distance-self.win_thre))
        if distance < self.win_thre:
            done = True
        if self.max_timestep > 0 and self.curr_timestep == self.max_timestep:
            done = True
        return self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.zeros(4,dtype=np.float32)
        self.state[0] = np.random.uniform(-self.X_lim,self.X_lim)
        self.state[2] = np.random.uniform(-self.Y_lim,self.Y_lim)
        self.curr_timestep = 0
        return self.state  
    
    # def render(self, mode='human', close=False):
    #     # Render the environment to the screen
    #     state = self.state
    #     x, x_dot, y, y_dot= state
    #     distance = math.sqrt((x-self.x_goal)**2+(y-self.y_goal)**2)
    #     self.print_particle_map(x,y)
    #     print("dist:%.2f x:%.2f y:%.2f x_dot:%.2f y_dot:%.2f " %(distance,x,y,x_dot,y_dot))



    
    # def print_particle_map(self, x,y):
    #     grid = [['.' for _ in range(int(2*self.Y_lim))] for _ in range(int(2*self.X_lim))]
    #     grid[int(x+self.X_lim)][int(y+self.Y_lim)] = 'o'

    #     c=grid
    #     #print(c)
    #     gridLen=len(grid)
    #     cyctime=len(grid[0])
    #     #print(cyctime) 
    #     i=0
    #     j=0
    #     for j in range(cyctime):
    #         if j < cyctime :
    #             for i in range(gridLen):
    #                 if i < gridLen :
    #                     print(c[i][j],end='')
    #                     i=i+1
    #         print()
    #         j=j+1

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
            #添加台车转换矩阵属性
            self.goaltrans = rendering.Transform()
            goal.add_attr(self.goaltrans)
            goalx = screen_width/2.0 # MIDDLE OF CART
            goaly = screen_height/2.0 # MIDDLE OF CART
            #设置平移属性
            self.goaltrans.set_translation(goalx, goaly)
            self.viewer.add_geom(goal)

            # 创建台车
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            # axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            cart.set_color(.8,.6,.4)

            #添加台车转换矩阵属性
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            #加入几何体台车
            self.viewer.add_geom(cart)
            #创建摆杆
            # l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            # pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            # pole.set_color(.8,.6,.4)
            #添加摆杆转换矩阵属性
            # self.poletrans = rendering.Transform(translation=(0, axleoffset))
            # pole.add_attr(self.poletrans)
            # pole.add_attr(self.carttrans)
            #加入几何体
            # self.viewer.add_geom(pole)
            #创建摆杆和台车之间的连接
            # self.axle = rendering.make_circle(polewidth/2)
            # self.axle.add_attr(self.poletrans)
            # self.axle.add_attr(self.carttrans)
            # self.axle.set_color(.5,.5,.8)
            # self.viewer.add_geom(self.axle)
            #创建台车来回滑动的轨道，即一条直线
            # self.track = rendering.Line((0,carty), (screen_width,carty))
            # self.track.set_color(0,0,0)
            # self.viewer.add_geom(self.track)

        if self.state is None: return None

        x,x_dot,y,y_dot = self.state
        cartx = x*scale+screen_width/2.0 # MIDDLE OF CART
        carty = y*scale+screen_height/2.0 # MIDDLE OF CART
        #设置平移属性
        self.carttrans.set_translation(cartx, carty)
        # self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

