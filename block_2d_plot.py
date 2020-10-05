import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from normflow_policy.envs import block2D


base_filename = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment'
exp_name = 'block2D_ppo_tf_garage'
SUCCESS_DIST = 0.025
plot_skip = 10
plot_traj = True
traj_skip = 2
GOAL = block2D.GOAL

epoch_num = 50
sample_num = 20
T = 200
tm = range(T)

for ep in range(0,epoch_num):
    if ((ep==0) or (not ((ep+1) % plot_skip))) and plot_traj:
        filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(ep) + '.pkl'
        infile = open(filename, 'rb')
        ep_data = pickle.load(infile)
        infile.close()

        epoch = ep_data['stats'].last_episode
        obs0 = epoch[0]['observations']
        act0 = epoch[0]['actions']
        rwd_s0 = epoch[0]['env_infos']['reward_dist']
        rwd_a0 = epoch[0]['env_infos']['reward_ctrl']
        pos = obs0[:,:2].reshape(T,1,2)
        vel = obs0[:,2:4].reshape(T,1,2)
        act = act0[:,:2].reshape(T,1,2)
        rwd_s = rwd_s0.reshape(T,1)
        rwd_a = rwd_a0.reshape(T,1)


        cum_rwd_s_epoch = 0
        cum_rwd_a_epoch = 0
        # cum_rwd_t_epoch = 0
        for sp in range(0,sample_num):
            if ((sp == 0) or (not ((sp + 1) % traj_skip))):
                sample = epoch[sp]
                p = sample['observations'][:,:2].reshape(T,1,2)
                v = sample['observations'][:, 2:4].reshape(T, 1, 2)
                a = sample['actions'][:,:2].reshape(T, 1, 2)
                rs = sample['env_infos']['reward_dist'].reshape(T, 1)
                cum_rwd_s_epoch = cum_rwd_s_epoch + np.sum(rs.reshape(-1))
                ra = sample['env_infos']['reward_ctrl'].reshape(T, 1)
                cum_rwd_a_epoch = cum_rwd_a_epoch + np.sum(ra.reshape(-1))
                pos = np.concatenate((pos,p), axis=1)
                vel = np.concatenate((vel, v), axis=1)
                act = np.concatenate((act, a), axis=1)
                rwd_s = np.concatenate((rwd_s, rs), axis=1)
                rwd_a = np.concatenate((rwd_a, ra), axis=1)

        fig = plt.figure()
        plt.title('Epoch '+str(ep))
        plt.axis('off')
        ax = fig.add_subplot(3, 4, 1)
        ax.set_title('s1')
        ax.plot(tm, pos[:, :, 0], color='g')
        ax = fig.add_subplot(3, 4, 2)
        ax.set_title('s2')
        ax.plot(tm, pos[:, :, 1], color='g')
        ax = fig.add_subplot(3, 4, 3)
        ax.set_title('sdot1')
        ax.plot(tm, vel[:, :, 0], color='b')
        ax = fig.add_subplot(3, 4, 4)
        ax.set_title('sdot2')
        ax.plot(tm, vel[:, :, 1], color='b')
        ax = fig.add_subplot(3, 4, 5)
        ax.set_title('a1')
        ax.plot(tm, act[:, :, 0], color='r')
        ax = fig.add_subplot(3, 4, 6)
        ax.set_title('a2')
        ax.plot(tm, act[:, :, 1], color='r')
        ax = fig.add_subplot(3, 4, 7)
        ax.set_title('rs')
        ax.plot(tm, rwd_s, color='m')
        ax = fig.add_subplot(3, 4, 8)
        ax.set_title('ra')
        ax.plot(tm, rwd_a, color='c')

# rewards_disc_rtn = np.zeros(epoch_num)
rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))

for ep in range(epoch_num):
    filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(ep) + '.pkl'
    infile = open(filename, 'rb')
    ep_data = pickle.load(infile)
    infile.close()
    epoch = ep_data['stats'].last_episode
    # rewards_disc_rtn[ep] = np.mean([epoch[s]['returns'][0] for s in range(sample_num)])
    rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
    for s in range(sample_num):
        pos_norm = np.linalg.norm(epoch[s]['observations'][:, :2], axis=1)
        success_mat[ep, s] = np.min(pos_norm)<SUCCESS_DIST

# for ep in range(epoch_num):
#     if ((ep == 0) or (not ((ep + 1) % plot_skip))):
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1, 2, 1)
#         ax2 = fig.add_subplot(1, 2, 2)
#         epoch = exp_log[ep]
#         for s in range(0,sample_num):
#             if (s == 0) or (not ((s + 1) % traj_skip)):
#                 p1 = epoch[s]['observations'][:, 0]
#                 p2 = epoch[s]['observations'][:, 1]
#                 d1 = p1 - GOAL[0]
#                 d2 = p2 - GOAL[1]
#                 # pos_norm = -np.linalg.norm(epoch[s]['observations'][:, :2] - GOAL, axis=1)
#                 # ax.plot(pos_norm)
#                 # ax1.plot(p1)
#                 # ax2.plot(p2)
#                 # ax1.plot(d1)
#                 # ax2.plot(d2)
#                 ax1.plot(d1**2)
#                 ax2.plot(d2**2)
#                 ax1.plot(d1 ** 2+d2**2)
#                 ax2.plot(d2 ** 2+d1**2)

success_stat = np.sum(success_mat, axis=1)*(100/sample_num)

fig = plt.figure()
plt.axis('off')
ax = fig.add_subplot(1, 2, 1)
ax.set_title('Progress')
ax.set_xlabel('Epoch')
ax.plot(rewards_undisc_mean, label='undisc. reward')
ax.legend()
ax = fig.add_subplot(1, 2, 2)
ax.set_ylabel('Succes rate')
ax.set_xlabel('Epoch')
ax.plot(success_stat)
ax.legend()

plt.show()

