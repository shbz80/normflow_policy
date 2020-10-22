import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
import torch
from normflow_policy.envs.block2D import T
import copy

font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SUCCESS_DIST = 0.025
epoch_start = 0
epoch_num = 100
tm = range(T)
sample_num = 15

base_filename = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment'
plt.rcParams["figure.figsize"] = (6,2)
####### rl progress ########33

# exps = ['block2d_nfppo_garage_7','block2D_ppo_torch_garage_5','block2d_nfppo_garage_9','block2D_ppo_torch_garage']
# color_list = ['b', 'g', 'm', 'c']
# legend_list = ['$NF-PPO-R$', '$PPO-R$', '$NF-PPO$', '$PPO$']

exps = ['block2d_nfppo_garage_e_3','block2D_ppo_torch_garage_5']
color_list = ['b', 'g']
legend_list = ['$NF-PPO$', '$PPO$']



fig1 = plt.figure()
plt.axis('off')
ax1 = fig1.add_subplot(1, 2, 1)
# ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
ax1.set_xlabel(r'Iteration')
ax1.set_ylabel(r'Reward')
ax1.set_xlim(0,105)
ax1.set_xticks(range(0, 100, 20))
ax1.set_ylim(-5.0e3,-1.5e3)
ax2 = fig1.add_subplot(1, 2, 2)
ax2.set_ylabel(r'Success \%')
ax2.set_xlabel('Iteration')
# ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
ax2.set_xlim(0,105)
ax2.set_xticks(range(0, 100, 20))
ax2.set_yticks([0,50,100])
ax2.set_ylim(0,105)

rewards_undisc_mean = np.zeros(epoch_num)
rewards_undisc_std = np.zeros(epoch_num)
success_mat = np.zeros((epoch_num, sample_num))

# block2d_nf = []
# for i in range(len(exps)):
#     for ep in range(epoch_num):
#         filename = base_filename + '/' + exps[i] + '/' + 'itr_' + str(ep) + '.pkl'
#         infile = open(filename, 'rb')
#         ep_data = pickle.load(infile)
#         infile.close()
#         epoch = ep_data['stats'].last_episode
#         rewards_undisc_mean[ep] = np.mean([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         rewards_undisc_std[ep] = np.std([np.sum(epoch[s]['rewards']) for s in range(sample_num)])
#         for s in range(sample_num):
#             # sample = np.min(np.absolute(epoch[s]['observations'][:,:3]), axis=0)
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :2], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     rl_progress = {'reward_mean':copy.deepcopy(rewards_undisc_mean),
#                    'reward_std': copy.deepcopy(rewards_undisc_std),
#                    'stats': copy.deepcopy(success_stat)}
#     block2d_nf.append(rl_progress)
#
# pickle.dump(block2d_nf, open("block2d_nf.p", "wb"))

block2d_nf = pickle.load( open( "block2d_nf.p", "rb" ) )
width = 3
offset = [1, 4]
for i in [0, 1]:
    rl_progress = block2d_nf[i]
    rewards_undisc_mean = rl_progress['reward_mean']
    rewards_undisc_std = rl_progress['reward_std']
    success_stat = rl_progress['stats']

    interval = 10
    # idx = i*width + offset[i]

    inds = np.array(range(0,epoch_num)[offset[i]::interval])
    # inds = [0] + list(inds)
    # del inds[-1]
    heights = rewards_undisc_mean[inds]
    yerr = rewards_undisc_std[inds]
    yerr[0] = 0
    idx = np.array(range(0, epoch_num)[i*width::interval])
    ax1.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
    # ax1.legend(prop={'size': font_size_3},frameon=False)
    ax1.legend(loc='upper left', bbox_to_anchor=(.5, 1.4), frameon=False, ncol=4, prop={'size': font_size_3})


    # inds = np.array(range(0,epoch_num)[idx::interval])
    # inds = [0] + list(inds)
    # del inds[-1]
    success = success_stat[inds]
    ax2.bar(idx, success,width, color=color_list[i])
    # ax2.legend(prop={'size': font_size_3},frameon=False)
plt.text(0.03, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.text(0.54, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
plt.subplots_adjust(left=0.1, bottom=0.23, right=.99, top=0.8, wspace=0.4, hspace=0.7)
fig1.savefig("blocks2d_nf_rl_progress.pdf")



# ############energy plots ###############
# plt.rcParams["figure.figsize"] = (6,2)
# def blocks_norm_flow_energy_x(x, x_star, x_dot, Phi, S, M=1):
#     with torch.no_grad():
#         phi_x = Phi(x) - Phi(x_star)
#     phi_x = phi_x.numpy().reshape(-1)
#     assert(phi_x.shape==(2,))
#     V = 0.5*phi_x.dot(S.dot(phi_x)) + 0.5*M*x_dot.dot(x_dot)
#     return V
#
# def blocks_norm_flow_energy_y(y, x_star, x_dot, Phi, S, M=1):
#     with torch.no_grad():
#         y = y - Phi(x_star)
#     y = y.numpy().reshape(-1)
#     assert(y.shape==(2,))
#     V = 0.5*y.dot(S.dot(y)) + 0.5*M*x_dot.dot(x_dot)
#     return V
# # init traj x
# X1_n = 50
# X2_n = 50
# X1 = np.linspace(-0.5, 0.5, X1_n)
# X2 = np.linspace(-0.5, 0.5, X2_n)
# X1, X2 = np.meshgrid(X1, X2)
# V = np.zeros((X1_n,X2_n))
#
# exp_name = 'block2d_nfppo_garage_19'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(0) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# Phi = ep_data['algo']._old_policy._module.phi
# K = np.eye(2)
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         X_dot = np.zeros(2)
#         V[i,j] = blocks_norm_flow_energy_x(X, torch.zeros(2).view(1,2), X_dot, Phi, K)
#
# levels = np.array(range(20))*0.02
# fig = plt.figure()
# # plt.axis('off')
# # plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
# ax1 = fig.add_subplot(1, 3, 1)
# ax1.set_xlim(-0.5, 0.5)
# ax1.set_xticks([-0.5,0,0.5])
# ax1.set_ylim(-0.5, 0.5)
# ax1.set_yticks([-0.5,0,0.5])
# ax1.contour(X1, X2, V, levels, label='Energy function', colors='g', alpha=0.7)
# # ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# # ax.plot_surface(S1, S2, F,
# #                 cmap='viridis', edgecolor='none',alpha=0.7)
# pos = epoch[0]['observations'][:, :2]
# ax1.plot(pos[:,0], pos[:,1], color='b',linewidth=3,label='x-space traj')
# ax1.scatter(0,0,color='r',marker='o',s=20, label='Goal')
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=1)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
# # ax1.set_title(r'\textbf{(c)}', position=(0.1,0.9), fontsize=font_size_2)
# ax1.legend(loc='upper left', bbox_to_anchor=(-0.15, 1.3),frameon=False,ncol=2,prop={'size': 10})
#
# # init traj y
# Y1_n = 50
# Y2_n = 50
# Y1 = np.linspace(-1.5, 0.5, Y1_n)
# Y2 = np.linspace(-0.5, 1.5, Y2_n)
# Y1, Y2 = np.meshgrid(Y1, Y2)
# V = np.zeros((Y1_n,Y2_n))
#
# for i in range(Y1_n):
#     for j in range(Y2_n):
#         Y = np.array([Y1[i, j], Y2[i, j]])
#         Y = Y.astype('float32')
#         Y = torch.from_numpy(Y).view(1,2)
#         X_dot = np.zeros(2)
#         V[i,j] = blocks_norm_flow_energy_y(Y, torch.zeros(2).view(1,2), X_dot, Phi, K)
#
# levels = np.array(range(35))*0.04
# # plt.axis('off')
# # plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
# ax2 = fig.add_subplot(1, 3, 3)
# ax2.set_xlim(-1.5, 0.2)
# ax2.set_xticks([-1.0, -0.5,0])
# ax2.set_ylim(-0.2, 1.5)
# ax2.set_yticks([0,0.5,1.0])
# ax2.contour(Y1, Y2, V, levels, label='Energy function', colors='c', alpha=0.7)
# # ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# # ax.plot_surface(S1, S2, F,
# #                 cmap='viridis', edgecolor='none',alpha=0.7)
# pos = epoch[0]['observations'][:, :2]
# pos = pos.astype('float32')
# pos = torch.from_numpy(pos)
#
# with torch.no_grad():
#     y_pos = Phi(pos).numpy()
#     goal = Phi(torch.zeros(2).view(1,2))
# goal = goal.numpy().reshape(-1)
# y_pos_init = y_pos - goal
# ax2.scatter(0,0,color='r',marker='o',s=20)
# ax2.set_xlabel(r'$y_1$',fontsize=font_size_2,labelpad=1)
# ax2.set_ylabel(r'$y_2$',fontsize=font_size_2, labelpad=0.2)
# # ax2.set_title(r'\textbf{(e)}', position=(0.1,0.9), fontsize=font_size_2)
#
# # final traj x
# X1_n = 50
# X2_n = 50
# X1 = np.linspace(-0.5, 0.5, X1_n)
# X2 = np.linspace(-0.5, 0.5, X2_n)
# X1, X2 = np.meshgrid(X1, X2)
# V = np.zeros((X1_n,X2_n))
#
# exp_name = 'block2d_nfppo_garage_20'
# filename = base_filename + '/' + exp_name + '/' + 'itr_' + str(100) + '.pkl'
# infile = open(filename, 'rb')
# ep_data = pickle.load(infile)
# infile.close()
# epoch = ep_data['stats'].last_episode
# Phi = ep_data['algo']._old_policy._module.phi
# K = np.eye(2)
# for i in range(X1_n):
#     for j in range(X2_n):
#         X = np.array([X1[i, j], X2[i, j]])
#         X = X.astype('float32')
#         X = torch.from_numpy(X).view(1,2)
#         X_dot = np.zeros(2)
#         V[i,j] = blocks_norm_flow_energy_x(X, torch.zeros(2).view(1,2), X_dot, Phi, K)
#
# levels = np.array(range(20))*0.2
# # plt.axis('off')
# # plt.legend(loc='upper right', bbox_to_anchor=(2., 1.),frameon=False,ncol=3)
# ax3 = fig.add_subplot(1, 3, 2)
# ax3.set_xlim(-0.5, 0.5)
# ax3.set_xticks([-0.5,0,0.5])
# ax3.set_ylim(-0.5, 0.5)
# ax3.set_yticks([-0.5,0,0.5])
# ax3.contour(X1, X2, V, levels, label='Energy function', colors='g', alpha=0.7)
# # ax.plot_wireframe(S1, S2, F, alpha=.7,label='Energy function')
# # ax.plot_surface(S1, S2, F,
# #                 cmap='viridis', edgecolor='none',alpha=0.7)
# pos = epoch[0]['observations'][:, :2]
# ax3.plot(pos[:,0], pos[:,1], color='b',linewidth=3)
# ax3.scatter(0,0,color='r',marker='o',s=20)
# ax3.set_xlabel(r'$x_1$',fontsize=font_size_2, labelpad=1)
# ax3.set_ylabel(r'$x_2$',fontsize=font_size_2, labelpad=0.2)
# pos = pos.astype('float32')
# pos = torch.from_numpy(pos)
# with torch.no_grad():
#     y_pos = Phi(pos).numpy()
#     goal = Phi(torch.zeros(2).view(1, 2))
# goal = goal.numpy().reshape(-1)
# y_pos = y_pos - goal
# ax2.plot(y_pos[:,0], y_pos[:,1], color='y',linewidth=3,label='trained')
# ax2.plot(y_pos_init[:,0], y_pos_init[:,1], color='m',linewidth=3,label='untrained')
# # ax3.set_title(r'\textbf{(d)}', position=(-0.3,-3), fontsize=font_size_2)
# plt.text(0.03, 0.05, '(c)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.36, 0.05, '(d)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.7, 0.05, '(e)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.5, 0.93, 'y-space traj:', fontweight='bold', fontsize=10, transform=plt.gcf().transFigure)
# plt.subplots_adjust(left=0.1, bottom=0.22, right=0.98, top=0.85, wspace=0.5, hspace=.1)
# ax2.legend(loc='upper left', bbox_to_anchor=(-0.62, 1.3),frameon=False,ncol=2,prop={'size': 10})
# fig.savefig("blocks2d_nf_init_traj_energy.pdf")