[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Udacity Deep Reinforcement Learning Nanodegree. Project 1: Continuous Control

## Introduction

This is the second project of the [Deep Reinforcement Learning Nanodegree (DRLND)](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). For this project, we train an agent to control a double-joint arm that can move to target locations. We choose to implement [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf).

![Trained Agent][image1]

We work with a Udacity adaptation of the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

For this project, two separate versions of the Unity environment are provided:
- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

We are solving the first agent of the environment with a single agent. The task is episodic, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Getting Started

1. Environment setup. This repository contains the environment for Linux. If you use a different operating system, please download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 1: One (1) Agent_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. If you downloaded the environment for your operating system, please decompress the file and place it in the `environment` directory of this repository.

3. Installing required dependencies. Please follow the instructions in the [DRLND GitHub repository](https://github.com/udacity/deep-reinforcement-learning#dependencies) to install all dependencies and set up your Python environment. These instructions can be found in `README.md` at the root of the repository. By following these instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to run this project.

## Instructions

Open the notebook [`Continuous_Control.ipynb`](Continuous_Control.ipynb), choose the 'drlnd' kernel and run all cells to train the PPO agent to solve the Reacher environment!
The notebook imports Python modules for agent and network definition, which can be found in the `agents` and `networks` directories.

## Report
For details on the implemented learning algorithm, including model architecture and hyperparameters, and on the achieved rewards please see the [Report](Report.md).