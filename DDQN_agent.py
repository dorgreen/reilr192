import gym
from gym import wrappers, logger
import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import resize
import random
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as kr


# - record actions taken (graph them?)
# - TensorBoard:
#       * TRAINING:
#         - score, reward, epsilone, steps in, loss (DEFINE!) PER EPISODE
#       * PLAYING:
#         - score, acations taken
#


WIDTH = 84
HEIGHT = 84


# does not inherit from object to make it lighter. could inherit later if needed
class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.state_size = WIDTH*HEIGHT*4
        self.epsilon = 0.95 # epsilon changes with "temperture", resets on each episode. takes about 900 runs
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.gamma = 0.85 # discount
        self.batch_size = 280
        self.learning_rate = 0.01
        self.memory = list()
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()



    def _huber_loss(self, target, prediction):
        # sqrt(1+error^2)-1
        error = prediction - target
        return kr.mean(kr.sqrt(1+kr.square(error))-1, axis=-1)

    # creates the network.
    # if using GPUs probably should be set here
    #      TODO: IMPLEMENT
    def create_model(self):

        with tf.device("/cpu:0"):
            inputs = Input(shape=(4, WIDTH, HEIGHT,))
            model = Conv2D(activation='relu', kernel_size=(8,8), filters=16,  strides=(4, 4),
                                  padding='same')(inputs)
            model = Conv2D(activation='relu', kernel_size=(4,4) ,filters=32, strides=(2, 2),
                                  padding='same')(model)
            model = Flatten()(model)
            # Last two layers are fully-connected
            model = Dense(activation='relu', units=256)(model)
            q_values = Dense(activation='linear', units=6)(model)
            m = Model(input=inputs, outputs=q_values)
            m.compile(loss=self._huber_loss,
                          optimizer=Adam(lr=self.learning_rate))

        return m

        #
        # model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dense(24, activation='relu'))
        # model.add(Dense(self.action_space.n, activation='linear'))
        # model.compile(loss='mse',
        #               optimizer=Adam(lr=self.learning_rate))
        # return model



    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

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
    #      TODO: add TensorBoard callback on model.predict for metrics report
    def replay(self, batch_size):
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_decay * self.epsilon )

        batch = random.sample(self.memory, batch_size)
        # for each transition in this batch, set target and fit to model accordingly


        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))

            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            loss = self.model.fit(state, target_f, epochs=1, verbose=0)

        return

    def load_weights_from_file(self, filename):
        self.model.load_weights(filename)

    def save_weights_to_file(self, filename):
        self.model.save_weights(filename, True)


"""turns a single frame from original format to the format in Q function"""
def resize_frame(ob):
    # TODO: if changing network input, change here
    return np.array(resize(rgb2gray(ob), (WIDTH, HEIGHT))).flatten()

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

    #
    return np.array(tuple(frames_buffer)).reshape([1, 4, WIDTH, HEIGHT])


def transform_reward(reward, done, lives_delta, episode=-1, action=-1):

    reward = -1 if reward == 0 else 10*reward
    if episode==-1:
        reward = -30 if lives_delta < 0 else reward

    reward = reward if not done else -50
    return reward

def train(agent, episode_count=1000, episode_length=5000):
    global frames_buffer

    ## Gain some experience..
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
        score = 0
        lives = 0


        ## for each episode we play according to agent for (at most) episode_length frames
        ## after X frames, update agent's model using the new data we gathered.
        for t in range(episode_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, info = env.step(action)
            prev_state = state
            state = obs_to_state(ob)
            score +=reward
            reward = transform_reward(reward,done, info['ale.lives']-lives,episode=i, action=action)
            lives = info['ale.lives']


            agent.remember(prev_state, action, reward, state, done)
            total_reward += reward

            env.render()
            if done:
                reset_frame_buffer()
                agent.update_target_model
                break

            # env.monitor can record video of some episodes. see capped_cubic_video_schedule

            # DEBUG ONLY
            if (t % 250 == 0):
                print("episode: {} step {}".format(i, t))


        # Train after each episode (for when we didn't quite make it to 2000 frames...)
        if agent.memory.__len__() > agent.batch_size:
            agent.replay(agent.batch_size)

        # Save every 25 runs
        if i % 25 == 0:
            agent.save_weights_to_file("./DDQN_Agent_saves/save{}.h5".format(i))
            agent.save_weights_to_file("./DDQN_Agent_saves/lastest.h5")

        # Close the env (and write monitor result info to disk if it was setup)
        print("EPISODE: {} SCORE: {} TOTAL REWARD {} epsilon {}".format(i,score, total_reward, agent.epsilon))
        env.close()

    agent.save_weights_to_file("./DDQN_Agent_saves/lastest.h5")
    agent.save_weights_to_file("./DDQN_Agent_saves/final.h5")

def play(agent, games=1, game_length=5000):
    for game in range(games):
        # Reset
        reset_frame_buffer()
        ob = env.reset()
        state = obs_to_state(ob)
        action = 0
        reward = 0
        done = 0
        agent.epsilon = agent.epsilon_min
        total_reward = 0
        lives = 0

        # play one game
        for t in range(game_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, info = env.step(action)
            state = obs_to_state(ob)
            reward = transform_reward(reward,done,info['ale.lives']-lives)
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
    agent.load_weights_from_file("./DDQN_Agent_saves/lastest.h5")
    train(agent, episode_count=1000)

    agent.load_weights_from_file("./DDQN_Agent_saves/lastest.h5")
    play(agent, games=10)
    # else:
    #     train(agent)
    #     play(agent, games=1)
