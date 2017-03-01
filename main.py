import gym
import tensorflow as tf
import os
import numpy as np
from numpy import random
from gym.spaces.box import Box
from DQN import experience_replay
from NET import *

def processState(state):
	return np.reshape(states,[21168])

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars/2]):
        op_holder.append(tfVars[idx+total_vars/2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars/2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

#import gym
#env = gym.make('Copy-v0')
#env.reset()
#env.render()

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
anneling_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network


total_steps = 0
myBuffer = experience_replay()

image = (imread("atari.png")[:,:,:3]).astype(float32)
image = image - mean(image)
xdim = image.shape[0:]
x = tf.placeholder(tf.float32, (None,) + xdim)
weights = load("weights_alexnet.py").item()


e = startE
stepDrop = (startE - endE)/anneling_steps
#Costructors of the Nets

targetQN = Q_Net()
mainQN = Q_Net()
targetQN.Preprocessing(weights,x)
#targetQN.Train()
targetQN.Predict()
mainQN.Preprocessing(weights,x)
mainQN.Predict()
mainQN.Train(targetQN)

init = tf.initialize_all_variables()


#targetops
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables,tau)

with tf.Session() as sess:
    sess.run(init)
    #
    for i in range(num_episodes):
            episodeBuffer = experience_replay()
            #Reset environment and get first new observation
            s = env.reset()
            s = processState(s)
            d = False
            rAll = 0
            j = 0
            #The Q-Network
            while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
                j+=1
                #Choose an action by greedily (with e chance of random action) from the Q-network
                if np.random.rand(1) < e or total_steps < pre_train_steps:
                    a = np.random.randint(0,2)
                else:
                    a = sess.run(mainQN.argmax,feed_dict={mainQN.scalarInput:[s]})[0]
                s1,r,d = env.step(a)
                s1 = processState(s1)
                total_steps += 1
                episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
                
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= stepDrop
                    
                    if total_steps % (update_freq) == 0:
                        trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                        #Below we perform the Double-DQN update to the target Q-values
                        Q1 = sess.run(mainQN.argmax,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                        Q2 = sess.run(targetQN.prob,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                        end_multiplier = -(trainBatch[:,4] - 1)
                        targetQ = trainBatch[:,2] + (y*Q2 * end_multiplier)
                        #Update the network with our target values.
                        _ = sess.run(mainQN.updateWeights, \
                            feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                        
                        updateTarget(targetOps,sess) #Set the target network to be equal to the primary network.
                rAll += r
                s = s1
                
                if d == True:

                    break
            
            myBuffer.add(episodeBuffer.buffer)

        
