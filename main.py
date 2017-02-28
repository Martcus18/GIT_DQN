#LIBRARIES IMPORTED

import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gym.spaces.box import Box
from DQN import experience_replay

#HYPERPARAMETERS

initial_epsilon = 1
final_epsilon = 0.1
decayment_steps = 1000
gamma   = 0.99
number_episodes = 10
max_lenght_episode = 50000
pre_train_steps = 100

#THE BATCH SIZE AND THE TRACE LENGHT ARE IMPORTANTS IN ORDER TO ACHIEVE A GOOD RESULT WITH GRADIENT DESCENDENT
#MINI-BATCH GRADIENT DESCENDENT APPROACH
batch_size = 4
update_freq = 5
update_target = 1000

#DA DEFINIRE MEGLI, CORRISPONDE AI PARAMETRI DEI LAYER FINALE CONVOLUTIVO
layer_size = 512

#Parameters inizialitation
j=0
BufferM = experience_replay()
k_list = []
r_list = []
total_steps=0


#CREATION OF THE PONG ENVIRONMENT

env = gym.make("Pong-v0")

step_reduction = (initial_epsilon - final_epsilon) / decayment_steps
UP = 1
DOWN = 2
STOP = 3
actions= np.array([UP,DOWN,STOP])

epsilon = initial_epsilon

for j in range(number_episodes):

 #PARAMETERS IN THE PREVIOUS TRAINING STEP
 previous_last_layer = (np.zeros([1,layer_size]),np.zeros([1,layer_size]))
 state = env.reset()
 newstate = state
 episode = []
 k = 0
 done = False
 total_reward = 0

#INNER LOOP
 while (k < max_lenght_episode):
     #env.render()
     #Epsilon-Greedy
     state = newstate
     #PER ORA 2, NON AVENDO UNA POLITICA ALTERNATIVA ( ARGMAX Q(S,A)) DA POTER USARE LO FACCIO ANDARE SEMPRE random
     if np.random.random() < 2 :
        #REDUCED THE NUMBER OF ACTIONS
        #action = env.action_space.sample() # your agent here (this takes random actions)
        #0,1 action  --> stop in the center of frame
        #2,4 action  --> go up
        #3,5 action  --> go down
        selected_action = np.random.choice(actions)
     newstate, reward, done, info = env.step(selected_action)

     #IF STATE NEED TO BE RESET, DONE BECOMES BOOLEAN TRUE AND STOP INNER LOOP
     if done == True :
        break

     total_reward = total_reward + reward
     episode.append(np.reshape(np.array([newstate,selected_action,reward,state,done]),[1,5]))

     #WAITING PRE_TRAIN_STEPS BEFORE STARTING TRAINING
     if (total_steps > pre_train_steps):

        #REDUCTION OF LEARNING RATE
        if epsilon > final_epsilon:
                    epsilon = epsilon - step_reduction

        if (total_steps % (update_freq * update_target)) == 0:
         #UPDATE TARGET Q FUNCTION, NOW DOING pass == NOOPERATION, TO BE UPDATED
           pass

        if (total_steps % update_freq) == 0:
          train_layer = (np.zeros([batch_size,layer_size]),np.zeros([batch_size,layer_size]))
          train_batch = BufferM.sample(batch_size)
         #NOW SHOULD BE FEED THE NEURAL NETWORKS WITH train_batch
         #BLA BLA BLA NEURAL NETWORK'S STUFF

     k=k+1
     #PARAMETERS IN THE ACTUAL LAYER
     actual_last_layer = previous_last_layer
     total_steps = total_steps + 1
     BufferM.add(episode)
     r_list.append(reward)
     k_list.append(k)
