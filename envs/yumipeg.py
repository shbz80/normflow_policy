import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

# GOAL 1
GOAL = np.array([-1.50337106, -1.24545874,  1.21963181,  0.46298941,  2.18633697,  1.51383283,
  0.57184653])
# obs in operational space
GOAL_CART = [ 0.46473501,  0.10293446,  0.10217953, -0.00858317,  0.69395054,  0.71995417,
              0.00499788,  0.,          0.,          0.,          0.,          0.,      0.]


# GOAL 2
# GOAL = np.array([-1.63688, -1.22777, 1.28612, 0.446995, 2.21936, 1.57011, 0.47748]) # goal

# INIT 1
INIT = np.array([-1.14, -1.21, 0.965, 0.728, 1.97, 1.49, 0.])
# obs in operational space
# [0.43412841 0.16020995 0.17902697 0.00230701 0.45355798 0.89116592
#  0.01015589 0.         0.         0.         0.         0.
#  0.        ]
# INIT = np.array([-1.14762187, -1.09474318,  0.72982478,  0.23000484,  1.7574765,   1.53849862,   0.4464969 ]) # init pos 2
# INIT = np.array([-1.60661071, -0.89088649,  1.0070413,   0.33067306,  1.8419217,   1.66532153, -0.06107046]) # init pos 3


# ACTION_SCALE = 1e-3
ACTION_SCALE = 1e-3
STATE_SCALE = 1
TERMINAL_SCALE = 100
H = 200
EXP_SCALE = 2.

class YumiPegEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'yumi_stable_vic_mjcf.xml')
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)
        self.reset_model()
        self.t = 0

    def step(self, a):
        pos = self.sim.data.qpos.flat[:7]
        dist = pos - GOAL
        reward_dist = -STATE_SCALE*np.linalg.norm(dist)
        reward_ctrl = -ACTION_SCALE * np.square(a).sum()
        reward = reward_dist + reward_ctrl

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        init_qpos = INIT
        init_qvel = np.zeros(7)
        self.set_state(init_qpos, init_qvel)
        # self.t = 0
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            # self.get_body_com("blocky")[:2],
        ])
