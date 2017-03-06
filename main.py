import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gym.spaces.box import Box
from DQN import experience_replay
from NET_0_12 import *



def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)



env = gym.make('Breakout-v0')


batch_size = 4 #How many experiences to use for each training step.
update_freq = 1000 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 0.9 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 100 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 10000 #The max allowed length of our episode.
#path = "./dqn" #The path to save our model to.
tau = 0.1 #Rate to update target network toward primary network


total_steps = 0
myBuffer = experience_replay()


epsilon = startE
stepDrop = (startE - endE)/anneling_steps


QTarget = Q_Net()
QMain = Q_Net()

init = tf.initialize_all_variables()
train = tf.trainable_variables()
#saver = tf.train.Saver()
sess = tf.Session()


total_steps = 0
reward = []
k=0
sess.run(init)
prob = 0
for i in range(num_episodes):
        episodeBuffer = experience_replay()
        #Reset environment and get first new observation
        state = env.reset()
        #env.render()
        done = False
        total_reward = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #print j
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if (np.random.rand(1) < epsilon) or (total_steps < pre_train_steps):
                action = env.action_space.sample()
            else:
                action = sess.run(QMain.argmax,feed_dict={QMain.input:[state]})[0]
            newstate,reward,done,info = env.step(action)
            if (done == True):
                reward -= 1.0
            total_steps += 1
            #print(total_steps)
            #RECORDING ONLY GOOD REWARD FOR pre_train_steps
            if total_steps < pre_train_steps:
                if reward == 0.:
                    if np.random.rand(1) > 0.95:
                        episodeBuffer.add(np.reshape(np.array([state,action,reward,newstate,done]),[1,5]))
                if (reward != 0.):
                    episodeBuffer.add(np.reshape(np.array([state,action,reward,newstate,done]),[1,5]))

            if total_steps >= pre_train_steps:
                #env.render()
                episodeBuffer.add(np.reshape(np.array([state,action,reward,newstate,done]),[1,5]))
                if epsilon > endE:
                        epsilon -= stepDrop
                if total_steps % (update_freq) == 0:
                        print(total_steps)
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        Q2 = np.zeros((batch_size,6))
                        QValue = np.zeros(batch_size)
                        MaxQ2 = np.zeros(batch_size)
                        targetQ = np.zeros(batch_size)
                        end_multiplier = -(trainBatch[:,4] - 1)
                        for l in range(batch_size):
                            Q2[l] = sess.run(QTarget.prob,feed_dict={QTarget.input:[trainBatch[:,3][l]]})
                            MaxQ2[l] = np.amax(Q2[l])
                            targetQ[l] = trainBatch[l,2] + (y*MaxQ2[l]*end_multiplier[l])
                            _,prob= sess.run([QMain.updateWeights,QMain.prob],feed_dict={QMain.input:[trainBatch[:,0][l]],QMain.target:[targetQ[l]], QMain.actions:[trainBatch[:,1][l]]})
                        print(total_reward)
                    #updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
            total_reward += reward
            state = newstate
            if done == True:
                break
        myBuffer.add(episodeBuffer.buffer)
