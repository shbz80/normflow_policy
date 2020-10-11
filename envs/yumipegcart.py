import os
import numpy as np
from yumikin.YumiKinematics import YumiKinematics
from normflow_policy.rewards import cart_rwd_func_1
from gym import utils
from gym.envs.mujoco import mujoco_env
from akro.box import Box
from garage import EnvSpec

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

T = 200
dA = 3
dO = 6
dJ = 7
D_rot = np.eye(3)*4

kin_params_yumi = {}
kin_params_yumi['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
kin_params_yumi['base_link'] = 'world'
# kin_params_yumi['end_link'] = 'left_tool0'
kin_params_yumi['end_link'] = 'left_contact_point'
kin_params_yumi['euler_string'] = 'sxyz'
kin_params_yumi['goal'] = GOAL

class YumiPegCartEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        fullpath = os.path.join(os.path.dirname(__file__), "mujoco_assets", 'yumi_stable_vic_mjcf.xml')
        self.initialized = False
        mujoco_env.MujocoEnv.__init__(self, fullpath, 5)
        self.kinparams = kin_params_yumi
        self.yumikin = YumiKinematics(self.kinparams)
        self.M_rot = np.diag(self.yumikin.get_cart_intertia_d(INIT))[3:]
        self.K_rot = 4*np.eye(3)
        self.D_rot = np.max(np.sqrt(np.multiply(self.M_rot,self.K_rot)))*np.eye(3)
        self.J_Ad_curr = None
        self.initialized = True
        self.action_space = Box(low=-5, high=5, shape=(dA,))
        self.observation_space = Box(low=-2, high=2.0, shape=(dO,))
        # self.reset_model()

    def step(self, a):
        if self.initialized:
            ex, jx = self._get_obs()
            f_t = a
            f_r = -np.matmul(self.K_rot,ex[3:6]) - np.matmul(self.D_rot,ex[9:])
            f = np.concatenate((f_t,f_r))
            reward, rewards = cart_rwd_func_1(ex, f)
            jtrq = self.J_Ad_curr.T.dot(f)
            assert (jtrq.shape == (dJ,))
            self.do_simulation(jtrq, self.frame_skip)
            done = False
            # return ex, reward, done, dict(reward_dist=np.sum(rewards[:-1]), reward_ctrl=rewards[-1])
            obs = np.concatenate((ex[:3],ex[6:9]))
            return obs, reward, done, dict({'er': ex[3:6],'erdot': ex[9:],'jx':jx, 'fr': f_r,'jt':jtrq})
        else:
            self.do_simulation(a, self.frame_skip)
            obs = self._get_obs()
            reward = None
            done = False
            return obs, reward, done, dict(reward_dist=None, reward_ctrl=None)

    def viewer_setup(self):
        # self.viewer.cam.trackbodyid = 0
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset(self):
        self.sim.reset()
        obs = self.reset_model()
        return obs

    def reset_model(self):
        init_qpos = INIT
        init_qvel = np.zeros(7)
        self.set_state(init_qpos, init_qvel)
        if self.initialized:
            del self.yumikin
            self.yumikin = YumiKinematics(self.kinparams)
            ex, jx = self._get_obs()
            return np.concatenate((ex[:3],ex[6:9]))
        else:
            return self._get_obs()

    def _get_obs(self):
        if self.initialized:
            jx = np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
            assert (jx.shape[0] == dJ * 2)
            q = jx[:dJ]
            q_dot = jx[dJ:]
            x_d_e, x_dot_d_e, J_Ad = self.yumikin.get_cart_error_frame_terms(q, q_dot)
            self.J_Ad_curr = J_Ad
            ex = np.concatenate((x_d_e, x_dot_d_e))
            # assert (ex.shape == (dO,))
            return ex, jx
        else:
            obs =  np.concatenate([
                self.sim.data.qpos.flat,
                self.sim.data.qvel.flat,
            ])
            return obs
