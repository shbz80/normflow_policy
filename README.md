# Learning Stable Normalizing-Flow Control for Robotic Manipulation
This repo is the code base for the paper _Learning Stable Normalizing-Flow Control for Robotic Manipulation_, Khader, S. A., Yin, H., Falco, P., & Kragic, D. (2020), IEEE International Conference on Robotics and Automation (ICRA). [[IEEE]](https://ieeexplore.ieee.org/document/9562071) [[arXiv]](https://arxiv.org/abs/2011.00072)

https://www.youtube.com/watch?v=Vl8HFq-lk94&t=1s

## Paper abstract
Reinforcement Learning (RL) of robotic manipulation skills, despite its impressive successes, stands to benefit from incorporating domain knowledge from control theory. One of the most important properties that is of interest is control stability. Ideally, one would like to achieve stability guarantees while staying within the framework of state-of-the-art deep RL algorithms. Such a solution does not exist in general, especially one that scales to complex manipulation tasks. We contribute towards closing this gap by introducing normalizing-flow control structure, that can be deployed in any latest deep RL algorithms. While stable exploration is not guaranteed, our method is designed to ultimately produce deterministic controllers with provable stability. In addition to demonstrating our method on challenging contact-rich manipulation tasks, we also show that it is possible to achieve considerable exploration efficiency–reduced state space coverage and actuation efforts– without losing learning efficiency.

## Prerequisites
* [garage](https://github.com/rlworkgroup/garage) Deep reinforcement learning toolkit
* [PyTorch](https://pytorch.org/) Deep learning framework
* [MuJoCo](https://mujoco.org/) Physics simulator

