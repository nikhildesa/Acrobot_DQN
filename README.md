# Acrobot_DQN

Reinforcement learning (RL) is an area of machine learning that focuses on how you,or how something, might act in an environment in order to maximize some given reward. Q learning is a model free reinforcement learning algorithm to
learn a policy telling an agent what action to
take under what circumstances.

# Overview
1. Learn - This implies we are not supposed
to hand-code any particular strategy but the
algorithm should learn by itself.
2. Policy - This is the result of the learning.
Given a State of the Environment , the Policy
will tell us how best to Interact with it to
maximize the Rewards .
3. Interact - This is nothing but the
“Actions” the algorithm should recommend
we take under different circumstances.
4. Environment - This is the black box the
algorithm interacts with. It is the game
which is supposed to be won. It is also the
world we live in. It’s the universe and all the
suns and the stars and everything else that
can influence the environment and it’s
reaction to the action taken.
5. Circumstances - These are the different
“States” the environment can be in.
6. Rewards - This is the goal. The purpose
of interacting with the environment is to
gain maximum rewards.

# Working
It is an Acrobot, i.e., a 2-link pendulum with only the second joint actuated. Initially, both links point downwards. The goal is to swing the end-effector and the upper link using DQN. Both links can swing freely and can pass by each other, i.e., they don’t collide when they have the same angle. The action is either applying +1, 0 or -1 torque on the joint between two pendulum links. 
