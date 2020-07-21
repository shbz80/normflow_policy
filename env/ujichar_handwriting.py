import gym
from gym import spaces
from gym.utils import seeding

import numpy as np

s_traj = np.array([ 0.85690839,  0.85743145,  0.8582298 ,  0.85890674,  0.85906555,
        0.85830951,  0.85623647,  0.85237651,  0.84621389,  0.83724237,
        0.82500077,  0.80904084,  0.78896952,  0.76448799,  0.73531617,
        0.70130457,  0.66239867,  0.61858411,  0.57002525,  0.51693983,
        0.45963147,  0.39856153,  0.33421587,  0.26719627,  0.19819504,
        0.12792477,  0.05718941, -0.01317795, -0.08233266, -0.14940724,
       -0.21353387, -0.27388472, -0.32966549, -0.38010508, -0.42455521,
       -0.4624104 , -0.49314688, -0.51641511, -0.53189674, -0.5394428 ,
       -0.53905671, -0.53078475, -0.51489864, -0.49175454, -0.46179414,
       -0.42565567, -0.38400998, -0.33763878, -0.28743075, -0.2342901 ,
       -0.1791942 , -0.12314987, -0.06716021, -0.0122188 ,  0.04068487,
        0.0906269 ,  0.13675133,  0.178227  ,  0.21437723,  0.24459265,
        0.26833992,  0.28528896,  0.2951489 ,  0.29778286,  0.29322512,
        0.2815445 ,  0.26300916,  0.23798081,  0.20687996,  0.1702963 ,
        0.12885223,  0.08324313,  0.03425143, -0.01733101, -0.07067536,
       -0.1249341 , -0.17927   , -0.23287853, -0.28496328, -0.33479225,
       -0.38171587, -0.42510601, -0.46448373, -0.49945134, -0.52967382,
       -0.55502672, -0.57543348, -0.59095592, -0.60184626, -0.60838745,
       -0.61103569, -0.61034861, -0.60691766, -0.60145443, -0.59469731,
       -0.58733874, -0.58000327, -0.57331   , -0.56787801, -0.56432639,
       -1.03701668, -1.03910357, -1.04233324, -1.04647665, -1.05130475,
       -1.05658848, -1.06209439, -1.06753429, -1.07258293, -1.07691347,
       -1.08019525, -1.08209671, -1.08229865, -1.08050293, -1.07641679,
       -1.06979298, -1.0604174 , -1.04809301, -1.03269961, -1.01414002,
       -0.99236035, -0.96738633, -0.93925672, -0.90807939, -0.87401615,
       -0.83724384, -0.79800867, -0.75657907, -0.71324441, -0.6683354 ,
       -0.62218823, -0.57514982, -0.52757611, -0.47982003, -0.43221633,
       -0.38509338, -0.33875833, -0.293473  , -0.24949062, -0.20701072,
       -0.16618457, -0.12714844, -0.08995892, -0.05464269, -0.02119414,
        0.01046663,  0.04043192,  0.06883732,  0.09586024,  0.12168437,
        0.1465234 ,  0.17060315,  0.19414887,  0.21738441,  0.24053251,
        0.26379411,  0.28734759,  0.31136312,  0.33595992,  0.36123501,
        0.38726141,  0.41404804,  0.44159158,  0.46984352,  0.49870513,
        0.52806857,  0.55777526,  0.58764285,  0.61747837,  0.64705835,
        0.67615384,  0.70453506,  0.73197128,  0.75823476,  0.78312384,
        0.80645   ,  0.82804074,  0.84777353,  0.86553666,  0.88125663,
        0.89490898,  0.90647823,  0.91600295,  0.92355115,  0.92920549,
        0.93309763,  0.93537005,  0.93618565,  0.93573531,  0.93421319,
        0.93182586,  0.92878715,  0.92531028,  0.92160587,  0.91788348,
        0.91432891,  0.91109287,  0.90832321,  0.90616781,  0.90477454]).reshape(2, 100).T
s_traj[:, 1] = -s_traj[:, 1]

class UJICharHandWritingEnv(gym.Env):
    def __init__(self):
        self.tar_traj = s_traj - s_traj[-1, :]  #make destination origin

        self.dt = .01
        self.m = 0.01
        self.t = 0
        
        self.viewer = None
        
        self.max_speed = 8
        self.max_force = 10.

        force_high = np.array([self.max_speed, self.max_speed])
        high = np.array([2.5, 2.5, self.max_speed, self.max_speed], dtype=np.float32)
        self.action_space = spaces.Box(
            low=(-1)*force_high,
            high=force_high,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=(-1)*high,
            high=high,
            dtype=np.float32
        )

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x       = self.state[:2]  
        x_dot   = self.state[2:]
        
        m = self.m
        dt = self.dt

        done = False

        u = np.clip(u, -self.max_force, self.max_force)
        self.last_u = u  # for rendering
        
        if self.t < self.tar_traj.shape[0]:
            target_x = self.tar_traj[self.t]
        else:
            target_x = self.tar_traj[-1]
        # target_x = self.tar_traj[-1]
        costs = np.sum((x-target_x)**2) + 0.01 * np.sum(u ** 2)

        #integrate system: semi-implicit 
        x_acc = u / m
        new_x_dot = x_dot + x_acc * dt
        new_x = x + new_x_dot * dt

        self.state = np.concatenate([new_x, new_x_dot])
        self.t+=1

        if self.t >= 100:
            done = True
        return self._get_obs(), -costs, done, {}

    def reset(self):
        self.state = np.concatenate([self.tar_traj[0] + self.np_random.randn(2)*0.05, np.zeros(2)])
        self.last_u = None
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        #fully observable...
        return np.array(self.state)

    def render(self, mode='human'):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1.2, 2.5, -1.2, 2.5)

            #reference trajectory, fixed
            ref_traj = rendering.make_polyline(self.tar_traj)
            ref_traj.set_linewidth(2.0)
            ref_traj.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(ref_traj)
            #dot agent
            dot_agent = rendering.make_circle(radius=0.05)
            dot_agent.set_color(1.0, 0.0, 0.0)

            self.dot_agent_pos = rendering.Transform()
            dot_agent.add_attr(self.dot_agent_pos)
            self.viewer.add_geom(dot_agent)


        # self.viewer.add_onetime(self.img)
        self.dot_agent_pos.set_translation(self.state[0], self.state[1])
        # if self.last_u:
        #     self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
