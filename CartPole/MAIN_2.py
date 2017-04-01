import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gym.spaces.box import Box
from Net2 import *
#from skimage.color import rgb2gray
import cv2
from collections import deque
import random







env = gym.make('CartPole-v0')

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.

y = .99 #Discount factor on the target Q-values
startE = 0.99 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 100000. #How many steps of training to reduce startE to endE.

num_episodes = 4000 #How many episodes of game environment to train network with.
pre_train_steps = 6000 #How many steps of random actions before training begins.
max_epLength = 1000 #The max allowed length of our episode.

target_freq = 1000
REPLAY_MEMORY_SIZE = 10000



stepDrop = (startE - endE)/anneling_steps

#tf.reset_default_graph()


QTarget = Q_Net()
QMain = Q_Net()

#train = tf.trainable_variables()
#saver = tf.train.Saver()

replay_memory = []




session = tf.Session()
epsilon = startE
saver = tf.train.Saver()
session.run(tf.initialize_all_variables())

checkpoint = tf.train.get_checkpoint_state("saved_networks")
if checkpoint and checkpoint.model_checkpoint_path:
  saver.restore(session, checkpoint.model_checkpoint_path)
  print("Successfully loaded:", checkpoint.model_checkpoint_path)
else:
  print("Could not find old network weights")

total_steps = 0
step_counts = []

session.run(QTarget.W_fc1.assign(QMain.W_fc1))
session.run(QTarget.W_fc2.assign(QMain.W_fc2))
session.run(QTarget.W_fc3.assign(QMain.W_fc3))
session.run(QTarget.b_fc1.assign(QMain.b_fc1))
session.run(QTarget.b_fc2.assign(QMain.b_fc2))
session.run(QTarget.b_fc3.assign(QMain.b_fc3))
'''
for episode in range(num_episodes):
  state = env.reset()
  steps = 0
  totalrew = 0
  for step in range(max_epLength):
    # Pick the next action and execute it
    #env.render()
    action = None
    if random.random() < epsilon:
      action = env.action_space.sample()
    else:
      q_values = session.run(QMain.out_fc3, feed_dict={QMain.input: [state]})
      action = q_values.argmax()
    if epsilon > endE:
        epsilon -= stepDrop

    obs, actual_reward, done, _ = env.step(action)

    if done:
      actual_reward = -1
    replay_memory.append((state, action, actual_reward, obs, done))
    if len(replay_memory) > REPLAY_MEMORY_SIZE:
      replay_memory.pop(0)
    state = obs

    if len(replay_memory) >= batch_size:
      minibatch = random.sample(replay_memory, batch_size)
      next_states = [m[3] for m in minibatch]
 
      q_values = session.run(QTarget.out_fc3, feed_dict={QTarget.input: next_states})
      max_q_values = q_values.max(axis=1)

      # Compute target Q values
      target_q = np.zeros(batch_size)
      target_action_mask = np.zeros((batch_size, 2), dtype=int)
      for i in range(batch_size):
        _, action, old_reward, _, done_ = minibatch[i]
        target_q[i] = old_reward
        if not done_:
          target_q[i] += y * max_q_values[i]
        target_action_mask[i][action] = 1

      states = [m[0] for m in minibatch]
      _= session.run(QMain.train_op, feed_dict={
        QMain.input: states, 
        QMain.target: target_q,
        QMain.actions: target_action_mask,
      })


    total_steps += 1
    totalrew += actual_reward
    steps += 1
    if total_steps % target_freq == 0:
        session.run(QTarget.W_fc1.assign(QMain.W_fc1))
        session.run(QTarget.W_fc2.assign(QMain.W_fc2))
        session.run(QTarget.W_fc3.assign(QMain.W_fc3))
        session.run(QTarget.b_fc1.assign(QMain.b_fc1))
        session.run(QTarget.b_fc2.assign(QMain.b_fc2))
        session.run(QTarget.b_fc3.assign(QMain.b_fc3))
    if done:
      break

  step_counts.append(steps) 
  mean_steps = np.mean(step_counts[-100:])
  print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}"
                                  .format(episode, total_steps, mean_steps))
  print("rew")
  print totalrew
  if num_episodes % 1000 == 0:
    saver.save(session, 'saved_networks/' + '-dqn')
'''
      
for i in range(num_episodes):
        old_state = env.reset()

        done = False
        total_reward = 0
        j = 0
        while j < max_epLength: 
            j+=1
            env.render()
            action = np.zeros(2)
            
            action_index = session.run(QMain.out_fc3,feed_dict = {QMain.input : [old_state]})
            action_index = action_index.argmax()
            action[action_index] = 1
            
            total_steps += 1

            new_state,reward,done,info = env.step(action_index)
            old_state = new_state
            total_reward += reward
            if done == True:
                print("PERSO")
                print j
                break
        print total_reward

