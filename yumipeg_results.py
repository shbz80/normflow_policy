import pickle
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib import rc
import torch
from normflow_policy.envs.yumipegcart import T, GOAL
from yumikin.YumiKinematics import YumiKinematics
import copy

font_size_1 = 12
font_size_2 = 14
font_size_3 = 10
plt.rcParams.update({'font.size': font_size_1})
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

yumikinparams = {}
yumikinparams['urdf'] = '/home/shahbaz/Software/yumi_kinematics/yumikin/models/yumi_ABB_left.urdf'
yumikinparams['base_link'] = 'world'
yumikinparams['end_link'] = 'left_contact_point'
yumikinparams['euler_string'] = 'sxyz'
yumikinparams['goal'] = GOAL
yumiKin = YumiKinematics(yumikinparams)
# GOAL = yumiKin.goal_cart
SUCCESS_DIST = .02

epoch_start = 0
epoch_num = 100
tm = range(T)
sample_num = 15

base_filename = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment'

##########rl progress###########
plt.rcParams["figure.figsize"] = (6,2)
base_filename = '/home/shahbaz/Software/garage36/normflow_policy/data/local/experiment'
# exps = ['yumipeg_nfppo_garage_9','yumipeg_ppo_garage_e','yumipeg_nfppo_garage_e_9','yumipeg_ppo_garage_e_1']
# color_list = ['b', 'g', 'm', 'c']
# legend_list = ['$NF-PPO-R$', '$PPO-R$', '$NF-PPO$', '$PPO$']

exps = ['yumipeg_nfppo_garage_e_14', 'yumipeg_ppo_garage_31']
color_list = ['b', 'g']
legend_list = ['$NF-PPO$', '$PPO$']

# fig1 = plt.figure()
# plt.axis('off')
# ax1 = fig1.add_subplot(1, 2, 1)
# # ax1.set_title(r'\textbf{(c)}', position=(-0.3, 0.94), fontsize=font_size_2)
# ax1.set_xlabel(r'Iteration')
# ax1.set_ylabel(r'Reward')
# ax1.set_xlim(0,105)
# ax1.set_xticks(range(0, 100, 20))
# # ax1.set_ylim(-1.0e4,-0.1e4)
# ax2 = fig1.add_subplot(1, 2, 2)
# ax2.set_ylabel(r'Success \%')
# ax2.set_xlabel('Iteration')
# # ax2.set_title(r'\textbf{(d)}', position=(-0.25, 0.94), fontsize=font_size_2)
# ax2.set_xlim(0,105)
# ax2.set_xticks(range(0, 100, 20))
# ax2.set_yticks([0,50,100])
# ax2.set_ylim(0,105)

# rewards_undisc_mean = np.zeros(epoch_num)
# rewards_undisc_std = np.zeros(epoch_num)
# success_mat = np.zeros((epoch_num, sample_num))
#
# yumipeg_nf = []
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
#             sample = np.min(np.linalg.norm(epoch[s]['observations'][:, :3], axis=1), axis=0)
#             # sample = epoch[s]['observations'][:, :6]
#             # success_mat[ep, s] = np.linalg.norm(sample) < SUCCESS_DIST
#             success_mat[ep, s] = sample < SUCCESS_DIST
#
#     success_stat = np.sum(success_mat, axis=1)*(100/sample_num)
#
#     rl_progress = {'reward_mean': copy.deepcopy(rewards_undisc_mean),
#                    'reward_std': copy.deepcopy(rewards_undisc_std),
#                    'stats': copy.deepcopy(success_stat)}
#     yumipeg_nf.append(rl_progress)
#
# pickle.dump(yumipeg_nf, open("yumipeg_nf.p", "wb"))

# yumipeg_nf = pickle.load( open( "yumipeg_nf.p", "rb" ) )
# width = 3
# offset = [0, 0]
# for i in [0,1]:
#     rl_progress = yumipeg_nf[i]
#     rewards_undisc_mean = rl_progress['reward_mean']
#     rewards_undisc_std = rl_progress['reward_std']
#     success_stat = rl_progress['stats']
#
#     # success_stat = np.sum(success_mat, axis=1) * (100 / sample_num)
#
#     interval = 10
#     # idx = i*width
#
#     inds = np.array(range(0,epoch_num)[offset[i]::interval])
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     heights = rewards_undisc_mean[inds]
#     yerr = rewards_undisc_std[inds]
#     yerr[0] = 0
#     idx = np.array(range(0, epoch_num)[i * width::interval])
#     ax1.errorbar(idx, heights, yerr=yerr, label=legend_list[i], color=color_list[i])
#     ax1.ticklabel_format(axis="y", style="sci", scilimits=(0,00))
#     # ax1.legend(prop={'size': font_size_3},frameon=False)
#     ax1.legend(loc='upper left', bbox_to_anchor=(0.5, 1.4), frameon=False, ncol=4, prop={'size': font_size_3})
#
#
#     # inds = np.array(range(0,epoch_num)[idx::interval])
#     # inds = [0] + list(inds)
#     # del inds[-1]
#     success = success_stat[inds]
#     ax2.bar(idx, success,width, color=color_list[i])
#     # ax2.legend(prop={'size': font_size_3},frameon=False)
# plt.text(0.03, 0.05, '(a)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.text(0.54, 0.05, '(b)', fontweight='bold', fontsize=font_size_2, transform=plt.gcf().transFigure)
# plt.subplots_adjust(left=0.1, bottom=0.23, right=.99, top=0.8, wspace=0.4, hspace=0.7)
# fig1.savefig("yumi_nf_rl_progress.pdf")

selected_samples = [0,1,2]
# # selected_samples = [0, 4, 12]
# plt.rcParams["figure.figsize"] = (3,3.75)
# fig = plt.figure()
# plt.axis('off')
# # nfppo_fixed_trained_det_samples, nfppo_rnd_trained_det_samples
# ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax1.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=0)
# ax1.set_ylabel(r'$x_2$',fontsize=font_size_2,labelpad=0)
# ax1.set_zlabel(r'$x_3$',fontsize=font_size_2,labelpad=0)
#
# for s in selected_samples:
#     # itr 0
#     base_number = 32
#     filename = base_filename + '/' + 'yumipeg_nfppo_garage_' + str(base_number+s) + '/' + 'itr_0' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     if selected_samples[0]==s:
#         ax1.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='b', linewidth=3, label='NF-PPO')
#     else:
#         ax1.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='b', linewidth=3)
#     ax1.scatter(sample[0, 0], sample[0, 1], sample[0, 2], color='b', marker='o', s=30)
#
#     base_number = 36
#     filename = base_filename + '/' + 'yumipeg_ppo_garage_' + str(base_number+s) + '/' + 'itr_0' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     if selected_samples[0] == s:
#         ax1.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='g', linewidth=3, label='PPO')
#     else:
#         ax1.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='g', linewidth=3)
#
# ax1.scatter(0,0,0,color='r',marker='o',s=50)
# ax1.set_yticklabels([])
# ax1.set_xticklabels([])
# ax1.set_zticklabels([])
# ax1.legend(loc='upper left', bbox_to_anchor=(0.25, 1.3),frameon=False,prop={'size': font_size_2},ncol=1)
# plt.subplots_adjust(left=-0.1, bottom=-0.15, right=1.05, top=.9, wspace=0.0, hspace=0.0)
#
# plt.rcParams["figure.figsize"] = (3,3.75)
# fig = plt.figure()
# plt.axis('off')
#
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# ax2.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=0)
# ax2.set_ylabel(r'$x_2$',fontsize=font_size_2,labelpad=0)
# ax2.set_zlabel(r'$x_3$',fontsize=font_size_2,labelpad=0)
#
# for s in selected_samples:
#     # itr 10
#     base_number = 37
#     filename = base_filename + '/' + 'yumipeg_nfppo_garage_' + str(base_number + s) + '/' + 'itr_9' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     ax2.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='b', linewidth=3)
#     if selected_samples[0] == s:
#         ax2.scatter(sample[0, 0], sample[0, 1], sample[0, 2], color='b', marker='o', s=30, label='Initial pos')
#     else:
#         ax2.scatter(sample[0, 0], sample[0, 1], sample[0, 2], color='b', marker='o', s=30)
#
#     base_number = 41
#     filename = base_filename + '/' + 'yumipeg_ppo_garage_' + str(base_number + s) + '/' + 'itr_9' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     ax2.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='g', linewidth=3)
#
# ax2.scatter(0,0,0,color='r',marker='o',s=50, label='Goal')
# ax2.set_yticklabels([])
# ax2.set_xticklabels([])
# ax2.set_zticklabels([])
# ax2.legend(loc='upper left', bbox_to_anchor=(0.25, 1.3),frameon=False,prop={'size': font_size_2},ncol=1)
# plt.subplots_adjust(left=-0.1, bottom=-0.15, right=1.05, top=.9, wspace=0.0, hspace=0.0)
#
# plt.rcParams["figure.figsize"] = (3,3)
# fig = plt.figure()
# plt.axis('off')
#
# ax3 = fig.add_subplot(1, 1, 1, projection='3d')
# ax3.set_xlabel(r'$x_1$',fontsize=font_size_2,labelpad=0)
# ax3.set_ylabel(r'$x_2$',fontsize=font_size_2,labelpad=0)
# ax3.set_zlabel(r'$x_3$',fontsize=font_size_2,labelpad=0)
#
# success_mat_nfppo = np.zeros(len(selected_samples))
# success_mat_ppo = np.zeros(len(selected_samples))
#
#
# i=0
# for s in selected_samples:
#
#     # itr 99
#     base_number = 42
#     filename = base_filename + '/' + 'yumipeg_nfppo_garage_' + str(base_number + s) + '/' + 'itr_99' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     x_g_min = np.min(np.linalg.norm(sample, axis=1), axis=0)
#     success_mat_nfppo[i] = x_g_min < SUCCESS_DIST
#     ax3.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='b', linewidth=3)
#     ax3.scatter(sample[0, 0], sample[0, 1], sample[0, 2], color='b', marker='o', s=30)
#
#     # itr 99
#     base_number = 46
#     filename = base_filename + '/' + 'yumipeg_ppo_garage_' + str(base_number + s) + '/' + 'itr_99' + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     epoch = ep_data['stats'].last_episode
#     sample = epoch[0]['observations'][:, :3]
#     x_g_min = np.min(np.linalg.norm(sample, axis=1), axis=0)
#     success_mat_ppo[i] = x_g_min < SUCCESS_DIST
#     ax3.plot3D(sample[:, 0], sample[:, 1], sample[:, 2], color='g', linewidth=3)
#     i +=1
#
# print('nfppo', success_mat_nfppo)
# print('ppo', success_mat_ppo)
# ax1.scatter(0,0,0,color='r',marker='o',s=50)
# ax2.scatter(0,0,0,color='r',marker='o',s=50)
# ax3.scatter(0,0,0,color='r',marker='o',s=50)
#
# ax3.set_yticklabels([])
# ax3.set_xticklabels([])
# ax3.set_zticklabels([])
# ax3.legend(loc='upper left', bbox_to_anchor=(0.25, 1.3),frameon=False,prop={'size': font_size_2},ncol=1)
# plt.subplots_adjust(left=-0.1, bottom=-0.15, right=1.05, top=.9, wspace=0.0, hspace=0.0)


# plt.rcParams["figure.figsize"] = (3,3)
# nfppo_state_dist_all = np.zeros((epoch_num, sample_num, T))
# nfppo_state_dist_last = np.zeros((epoch_num,sample_num))
# ppo_state_dist_all = np.zeros((epoch_num, sample_num, T))
# ppo_state_dist_last = np.zeros((epoch_num,sample_num))
# for i in range(0, epoch_num):
#     filename = base_filename + '/' + exps[0] + '/' + 'itr_' + str(i) + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     nfppo_epoch = ep_data['stats'].last_episode
#
#     filename = base_filename + '/' + exps[1] + '/' + 'itr_' + str(i) + '.pkl'
#     infile = open(filename, 'rb')
#     ep_data = pickle.load(infile)
#     infile.close()
#     ppo_epoch = ep_data['stats'].last_episode
#     for s in range(sample_num):
#         nfppo_state_dist_all[i][s] = np.linalg.norm(nfppo_epoch[s]['observations'][:, :3], axis=1).reshape(-1)
#         nfppo_state_dist_last[i][s] = np.linalg.norm(nfppo_epoch[s]['observations'][:, :3], axis=1)[-1]
#         ppo_state_dist_all[i][s] = np.linalg.norm(ppo_epoch[s]['observations'][:, :3], axis=1).reshape(-1)
#         ppo_state_dist_last[i][s] = np.linalg.norm(ppo_epoch[s]['observations'][:, :3], axis=1)[-1]
#
# itr_state_dist = 10
# nfppo_state_dist_all = nfppo_state_dist_all[:itr_state_dist,:,:].reshape(-1)
# nfppo_state_dist_last = nfppo_state_dist_last[:itr_state_dist,:].reshape(-1)
# ppo_state_dist_all = ppo_state_dist_all[:itr_state_dist,:,:].reshape(-1)
# ppo_state_dist_last = ppo_state_dist_last[:itr_state_dist,:].reshape(-1)
#
# fig = plt.figure()
# # plt.axis('off')
# ax4 = fig.add_subplot(1, 1, 1)
# # ax1.set_title(r'\textbf{(a)}')
# # data = [dist_ours, dist_vices, last_dist_ours, last_dist_vices]
# data = [nfppo_state_dist_all, ppo_state_dist_all, nfppo_state_dist_last, ppo_state_dist_last]
# # bp = ax1.boxplot(data, patch_artist = False, showfliers=False, whis=(0,100),vert=False)
# bp = ax4.boxplot(data, patch_artist = False, showfliers=False,  whis=(0,100), vert=False)
# for median in bp['medians']:
#     median.set(color ='blue',
#                linewidth = 1)
# ax4.set_yticklabels(['All pos\n(NF-PPO)','All pos\n(PPO)','Final pos\n(NF-PPO)','Final pos\n(PPO)'])
# ax4.set_xlabel('m',labelpad=1)
# # plt.subplots_adjust(left=0.0, bottom=0.04, right=1., top=1., wspace=0.0, hspace=0.0)
# plt.show(block=True)

# # print comparison table
# # std_l = [0.5, 1.0, 1.5, 2.0, 3.0]
# std_l = [1.0, 1.5, 2.0, 3.0]
# # exp_l = ['yumipeg_nfppo_garage_e_16','yumipeg_nfppo_garage_e_14','yumipeg_nfppo_garage_e_15','yumipeg_nfppo_garage_e_8']
# exp_l = ['yumipeg_nfppo_garage_e_14','yumipeg_nfppo_garage_e_15', 'yumipeg_nfppo_garage_e_18','yumipeg_nfppo_garage_e_8']
# trq_thresh = 8.
# itr_state_dist = 10
# print('NFPPO')
# assert(len(std_l)==len(exp_l))
# for i in range(len(exp_l)):
#     print('Init std. :', std_l[i])
#     state_dist_all = np.zeros((epoch_num, sample_num, T))
#     trq_dist_all = np.zeros((epoch_num, sample_num, T))
#     for ep in range(epoch_num):
#         filename = base_filename + '/' + exp_l[i] + '/' + 'itr_' + str(ep) + '.pkl'
#         infile = open(filename, 'rb')
#         ep_data = pickle.load(infile)
#         infile.close()
#         epoch = ep_data['stats'].last_episode
#
#         for s in range(sample_num):
#             state_dist_all[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1).reshape(-1)
#             trq_dist_all[ep][s] = np.linalg.norm(epoch[s]['env_infos']['jt'].reshape(T,7), axis=1).reshape(-1)
#
#     state_dist_all = state_dist_all[:itr_state_dist, :, :].reshape(-1)
#     print('state dist 10 itr mean:', np.mean(state_dist_all))
#     print('state dist 10 itr std:', np.std(state_dist_all))
#
#     trq_dist_all = trq_dist_all[:itr_state_dist, :, :].reshape(-1)
#     # trq_dist_all = trq_dist_all.reshape(-1)
#     # temp = np.zeros_like(trq_dist_all)
#     # temp[trq_dist_all > trq_thresh] = 1
#     # print('trq_error_cnt',np.sum(temp))
#     # print('trq_error', trq_dist_all[trq_dist_all > trq_thresh])
#     print('trq dist mean:', np.mean(trq_dist_all))
#     print('trq dist std:', np.std(trq_dist_all))
#
# print('PPO')
# exp_l = ['yumipeg_ppo_garage_31','yumipeg_ppo_garage_32', 'yumipeg_ppo_garage_51','yumipeg_ppo_garage_e_3']
# assert(len(std_l)==len(exp_l))
# for i in range(len(exp_l)):
#     print('Init std. :', std_l[i])
#     state_dist_all = np.zeros((epoch_num, sample_num, T))
#     trq_dist_all = np.zeros((epoch_num, sample_num, T))
#     for ep in range(epoch_num):
#         filename = base_filename + '/' + exp_l[i] + '/' + 'itr_' + str(ep) + '.pkl'
#         infile = open(filename, 'rb')
#         ep_data = pickle.load(infile)
#         infile.close()
#         epoch = ep_data['stats'].last_episode
#
#         for s in range(sample_num):
#             state_dist_all[ep][s] = np.linalg.norm(epoch[s]['observations'][:, :3], axis=1).reshape(-1)
#             trq_dist_all[ep][s] = np.linalg.norm(epoch[s]['env_infos']['jt'].reshape(T,7), axis=1).reshape(-1)
#
#     state_dist_all = state_dist_all[:itr_state_dist, :, :].reshape(-1)
#     print('state dist 10 itr mean:', np.mean(state_dist_all))
#     print('state dist 10 itr std:', np.std(state_dist_all))
#
#     trq_dist_all = trq_dist_all[:itr_state_dist, :, :].reshape(-1)
#     # trq_dist_all = trq_dist_all.reshape(-1)
#     print('trq dist mean:', np.mean(trq_dist_all))
#     print('trq dist std:', np.std(trq_dist_all))

# NFPPO
# Init std. : 1.0
# state dist 10 itr mean: 0.09820663274252514
# state dist 10 itr std: 0.053675057074935584
# trq dist mean: 0.916529701821852
# trq dist std: 1.1335289511319313
# Init std. : 1.5
# state dist 10 itr mean: 0.10389427025541609
# state dist 10 itr std: 0.05585529112427029
# trq dist mean: 1.3579825657529798
# trq dist std: 1.1008279356420847
# Init std. : 2.0
# state dist 10 itr mean: 0.1213153694851613
# state dist 10 itr std: 0.08532846350159227
# trq dist mean: 2.0438826926889258
# trq dist std: 5.086045349696357
# Init std. : 3.0
# state dist 10 itr mean: 0.1379400219167817
# state dist 10 itr std: 0.07234034717091695
# trq dist mean: 2.6832340316426295
# trq dist std: 2.0983915922179763
# PPO
# Init std. : 1.0
# state dist 10 itr mean: 0.1719453277582991
# state dist 10 itr std: 0.09612921521045635
# trq dist mean: 0.9051226806711031
# trq dist std: 1.1424942626826644
# Init std. : 1.5
# state dist 10 itr mean: 0.17591549362830636
# state dist 10 itr std: 0.10486030783980918
# trq dist mean: 1.3380059233805217
# trq dist std: 1.1531517730441465
# Init std. : 2.0
# state dist 10 itr mean: 0.18028449522730616
# state dist 10 itr std: 0.11651441927571515
# trq dist mean: 1.7832212099232034
# trq dist std: 3.4267205169406596
# Init std. : 3.0
# state dist 10 itr mean: 0.1976330425933497
# state dist 10 itr std: 0.1258239941909964
# trq dist mean: 2.643024638789287
# trq dist std: 1.680970066738028

# deterministic exec results
# nfppo [1. 1. 1.]
# ppo [1. 0. 1.]