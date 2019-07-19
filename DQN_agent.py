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


"""TODOs:
- set size as property of network
- change to conv2d

"""

WIDTH = 210
HEIGHT = 160

# does not inherit from object to make it lighter. could inherit later if needed
class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 1.0 # epsilon changes with "temperture", resets on each episode
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.85 # discount
        self.batch_size = 512
        self.learning_rate = 0.001
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
            action_vector = self.model.predict(state)
            return np.argmax(action_vector)

        else:
            return random.randrange(start=0, stop=self.action_space.n)

    # add this transition to memory
    #      TODO: Add feature that saves image of state once in a while for debug
    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= 2000:
            self.memory.pop()
        self.memory.append((state, action,reward,next_state,done))
        return

    # train model with the new data in memory
    #      TODO: IMPLEMENT
    #      TODO: add TensorBoard callback on model.predict for metrics report
    def replay(self, batch_size):
        # update epsilon
        if self.epsilon_min > self.epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

        batch = random.sample(self.memory, batch_size)

        # for each transition in this batch, set target and fit to model accordingly

        #states = list(map(lambda x: x[0], batch))
        #targets = self.model.predict(states, batch_size, 1)

        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))

            target_f = self.model.predict(state)
            target_f[0][action] = target
            loss = self.model.fit(state, target_f, epochs=1, verbose=0)

        return

    #      TODO: TEST
    def load_weights_from_file(self, filename):
        self.model.load_weights(filename)

    #      TODO: TEST
    def save_weights_to_file(self, filename):
        self.model.save_weights(filename, True)


"""turns a single frame from original format to the format in Q function"""
def resize_frame(ob):
    # TODO: if changing network input, change here
    return np.array(rgb2gray(ob)).flatten()

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

    # flatten from 4 * 33600 to 134400 as needed
    return np.array(tuple(frames_buffer)).flatten().reshape([1, WIDTH*HEIGHT*4])

def transform_reward(reward, done, time):
    reward = reward if not done else -20
    reward = -(time / 500) if reward == 0 else reward
    return reward

def train(agent, episode_count=100, episode_length=5000):
    global frames_buffer
    agent.epsilon = 1.0

    for i in range(episode_count):
        ## Original obervation is an nd array of (210, 160, 3) , such that H * W * rgb
        reset_frame_buffer()
        ob = env.reset()
        state = obs_to_state(ob)
        prev_state = state
        action = 0
        reward = 0
        done = False

        total_reward = 0


        ## for each episode we play according to agent for (at most) episode_length frames
        ## after X frames, update agent's model using the new data we gathered.
        for t in range(episode_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, _ = env.step(action)
            prev_state = state
            state = obs_to_state(ob)
            reward = transform_reward(reward,done,t)

            agent.remember(prev_state, action, reward, state, done)
            total_reward += reward

            trained = False

            env.render()
            if done:
                reset_frame_buffer()
                break

            # env.monitor can record video of some episodes. see capped_cubic_video_schedule

            # DEBUG ONLY
            if (t % 250 == 0):
                print("episode: {} step {}".format(i, t))

            if (t % 2000 == 0 and t > 0):
                agent.replay(agent.batch_size)
                trained = True

        # Train after each episode (for when we didn't quite make it to 2000 frames...)
        if (not trained) and agent.memory.__len__() > agent.batch_size:
            agent.replay(agent.batch_size)

        # Save every 25 runs
        if i % 25 == 0:
            agent.save_weights_to_file("./save{}.h5".format(i))
            agent.save_weights_to_file("./lastest.h5")

        # Close the env (and write monitor result info to disk if it was setup)
        print("EPISODE: {} TOTAL REWARD {}".format(i, total_reward))
        env.close()

    agent.save_weights_to_file("./lastest.h5")
    agent.save_weights_to_file("./final.h5")

def play(agent, games=1, game_length=10000):
    for game in range(games):
        # Reset
        reset_frame_buffer()
        ob = env.reset()
        state = obs_to_state(ob)
        action = 0
        reward = 0
        done = 0
        agent.epsilon = 0 # agent.min_epsilon
        total_reward = 0

        # play one game
        for t in range(game_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, _ = env.step(action)
            state = obs_to_state(ob)
            reward = transform_reward(reward,done,t)
            total_reward += reward

            env.render()
            if done:
                reset_frame_buffer()
                print("game: {} total reward: {}".format(game, total_reward))
                break


if __name__ == '__main__':

    # TODO: set args to know if we should train or load from disk and just play according to model
    # TODO: set training via GPU?
    # TODO: add TensorBoard for logging and reporting. [could be used without TF scoping?]


    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('DemonAttack-v0')


    ## Setup monitor if needed
    ## possible to use tempfile.mkdtemp().
    # outdir = '/tmp/random-agent-results'
    # env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = DQNAgent(env.action_space)
    train(agent, episode_count=10)

    agent.load_weights_from_file("./final.h5")
    play(agent, games=10)
    # else:
    #     train(agent)
    #     play(agent, games=1)
