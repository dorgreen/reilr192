import gym
from gym import wrappers, logger
import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
import random
import asyncio.queues
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# does not inherit from object to make it lighter. could inherit later if needed
class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 1.0 # epsilon changes with "temperture"
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.batch_size = 150
        self.learning_rate = 0.02
        self.memory = list()
        self.model = self.create_model()


    # creates the network.
    # if using GPUs probably should be set here
    #      TODO: IMPLEMENT
    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=WIDTH*HEIGHT*4, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def get_action(self, state, reward, done):
        """epsilon-greedy"""
        if random.random() > self.epsilon:
            action_vector = self.predict(state)
            return tf.arg_max(action_vector)

        else:
            return random.randrange(start=0, stop=self.action_space.n)

    # add this transition to memory
    #      TODO: Add feature that saves image of state once in a while for debug
    def remember(self, state, action, reward, next_state, done):
        if(len(self.memory) >= 2000):
            self.memory.pop()
        self.memory.append((state, action,reward,next_state,done))
        return

    # train model with the new data in memory
    #      TODO: IMPLEMENT
    def replay(self, batch_size):
        # update epsilon
        if self.epsilon_min > self.epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

        batch = random.sample(self.memory, batch_size)
        # for each transition in this batch, set target and fit to model accordingly
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array(state).flatten())
            if not done:
                a = self.model.predict(np.array(next_state).flatten())[0]
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * t[np.argmax(a)]
            else:
                target[0][action] = reward
            self.model.fit(state, target, epochs=1, verbose=0)

        return

    #      TODO: IMPLEMENT
    def load_weights_from_file(self, filename):
        return self.model.set_weights(filename)

    #      TODO: IMPLEMENT
    def save_weights_to_file(self, filename):
        return self.model.save_weights(filename, True)

WIDTH = 210
HEIGHT = 160

"""turns a single frame from original format to the format in Q function"""
def resize_frame(ob):
    # TODO: if changing network input, change here
    return rgb2gray(ob)

temp_env = gym.make('DemonAttack-v0')
empty_frame = resize_frame(temp_env.reset())
temp_env.close()
frames_buffer = list()

def reset_frame_buffer():
    frames_buffer.clear()
    for i in range(4):
        frames_buffer.append(empty_frame)
    return


"""turns observation ob from env to state as used by agent"""
def obs_to_state(ob):
    # TODO: if changing network input, change here
    this_frame = resize_frame(ob)
    frames_buffer.pop(0) # oldest frame discarded
    frames_buffer.append(this_frame)

    # return ob;
    return tuple(frames_buffer)

if __name__ == '__main__':

    # TODO: set args to know if we should train or load from disk and just play according to model
    # TODO: set training via GPU?
    # TODO: add TensorBoard for logging and reporting. can I do it without using TF scoping?


    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('DemonAttack-v0')

    # possible tempfile.mkdtemp().

    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env.action_space)

    episode_count = 100
    episode_length = 5000
    reward = 0
    done = False

    for i in range(episode_count):
        ## Original obervation is an nd array of (210, 160, 3) , such that H * W * rgb
        reset_frame_buffer()
        ob = env.reset()
        state = obs_to_state(ob)
        prev_state = state
        action = 0

        ## for each episode we play according to agent for (at most) episode_length frames
        ## after X frames, update agent's model using the new data we gathered.
        for t in range(episode_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, _ = env.step(action)
            prev_state = state
            state = obs_to_state(ob)
            agent.remember(prev_state, action, reward, state, done)

            env.render()
            if done:
                break

            # env.monitor can record video of some episodes. see capped_cubic_video_schedule

            if(t % 1000 == 0 and t>0):
                agent.replay(agent.batch_size)

        # Train after each episode (for when we didn't quite make it to 1000 frames...)
        if agent.memory.__len__() > agent.batch_size: agent.replay(agent.batch_size)

    # Close the env and write monitor result info to disk
    env.close()