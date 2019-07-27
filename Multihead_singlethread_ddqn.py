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
import os



# - record actions taken (graph them?)
# - TensorBoard:
#       * TRAINING:
#         - score, reward, epsilone, steps in, loss (DEFINE!) PER EPISODE
#       * PLAYING:
#         - score, acations taken
#


WIDTH = 84
HEIGHT = 84
save_file_path = "MultiHead_DQN_Agent_saves/"


def transform_reward(reward, done, lives_delta, episode=-1, action=-1):
    reward = -1 if reward == 0 else 10 * reward
    if episode == -1:
        reward = -10 if lives_delta < 0 else reward
    else:
        reward = max(-10, -0.5 * episode) if lives_delta < 0 else reward
    reward = reward if not done else -20
    return reward


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
    frames_buffer.pop(0)  # oldest frame discarded
    frames_buffer.append(this_frame)

    #
    return np.array(tuple(frames_buffer)).reshape([1, 4, WIDTH, HEIGHT])


def get_action_from_network(network, state, epsilon):
    """epsilon-greedy"""
    if random.random() > epsilon:
        action_vector = network.predict(state)
        return np.argmax(action_vector)

    else:
        return random.randrange(start=0, stop=6)


def replay_network(network, target_network, batch, gamma=0.85):
    print("train")
    target_network.set_weights(network.get_weights())
    # for each transition in this batch, set target and fit to model accordingly

    for state, action, reward, next_state, done in batch:
        target = reward

        if not done:
            target = (reward + gamma * np.amax(network.predict(next_state)))

        # or should it be target_f[0][action] - target ?
        loss = target - reward

        target_f = target_network.predict(state)
        target_f[0][action] = target
        loss = network.fit(state, target_f, epochs=1, verbose=0)


class MultiHeadDQNAgent:
    # TODO: maybe add bootstrap map as a parameter
    def __init__(self, action_space, num_of_agents, epsilon=0.95, agents_epsilon=0.95):
        self.action_space = action_space
        self.agents_num = num_of_agents
        self.state_size = WIDTH * HEIGHT * 4
        self.epsilon = epsilon  # epsilon changes with "temperture", resets on each episode. takes about 900 runs
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.993
        self.gamma = 0.9  # discount
        self.learning_rate = 0.0001
        self.networks = self.create_model(num_of_agents)
        self.target_networks = self.create_model(num_of_agents)

        networks = zip(self.networks, self.target_networks)
        for net, target_net in networks:
            target_net.set_weights(net.get_weights())

        self.memory = list()

        # creates the agents.
        # if using GPUs probably should be set here

    def huber_loss(self, target, prediction):
        error = prediction - target
        return kr.mean(kr.sqrt(1 + kr.square(error)) - 1, axis=-1)

    def create_single_network(self):
        inputs = Input(shape=(4, WIDTH, HEIGHT,))
        model = Conv2D(activation='relu', kernel_size=(4, 8), filters=32, strides=(4, 4),
                       padding='same', kernel_initializer='random_uniform')(inputs)
        model = Conv2D(activation='relu', kernel_size=(3,3), filters=64, strides=(1, 1),
                       padding='same', kernel_initializer='random_uniform')(model)
        model = Flatten()(model)
        # Last two layers are fully-connected
        model = Dense(activation='relu', units=512, kernel_initializer='random_uniform')(model)
        q_values = Dense(activation='linear', units=6)(model)
        m = Model(input=inputs, outputs=q_values)
        m.compile(loss=self.huber_loss,
                  optimizer=Adam(lr=self.learning_rate))
        return m

    # Networks is a list of networks
    def create_model(self, num_of_agents):
        networks = list()
        for i in range(num_of_agents):
            networks.append(self.create_single_network())
        return networks


    def get_action(self, state, reward, done):
        """epsilon-greedy"""
        # Let all models vote
        # Other options: action with best sum of values, the action with the highest value on some network
        if random.random() > self.epsilon:
            with tf.device("/cpu:0"):
                actions = bytearray(6)
                for net in self.networks:
                    action = get_action_from_network(net, state, 0)
                    actions[action] += 1
                return np.argmax(actions)

        else:
            return random.randrange(start=0, stop=self.action_space.n)

        # add this transition to memory
        #      TODO: Add feature that saves image of state once in a while for debug

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= 1000000:
            self.memory.pop()
        self.memory.append((state, action, reward, next_state, done))
        return

        # train model with the new data in memory
        #      TODO: add TensorBoard callback on model.predict for metrics report

    def replay(self):
        print("Replay")

        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_decay * self.epsilon)

        # Get a batch for each ddqn agent:
        size = (int) (len(self.memory) / (self.agents_num * 2))
        networks = zip(self.networks, self.target_networks)
        with tf.device("gpu:0"):
            for net, target_net in networks:
                batch = random.choices(self.memory, k=size)
                replay_network(net, target_net, batch, gamma=self.gamma)

        return

    def load_weights_from_file(self, filename: str):
        index = 0
        for model in self.networks:
            model.load_weights(filename + "{}.h5".format(index))
            index += 1
        return

    def save_weights_to_file(self, filename):
        index = 0
        for model in self.networks:
            model.save_weights(filename + "{}.h5".format(index), True)
            index += 1
        return


    def train(self, episode_count=1000, episode_length=5000):
        ## Gain some experience..
        for i in range(episode_count):
            # Reset items
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

            # Select a random agent for this episode
            selected_network = self.networks[random.randrange(start=0, stop=self.agents_num)]

            ## for each episode we play according to head_agent for (at most) episode_length frames
            ## for each agent in self.agents, record each transition acording to it's probability
            for t in range(episode_length):
                action = get_action_from_network(selected_network, state, self.epsilon)
                ob, reward, done, info = env.step(action)
                prev_state = state
                state = obs_to_state(ob)
                score += reward
                reward = transform_reward(reward, done, info['ale.lives'] - lives, episode=i, action=action)
                lives = info['ale.lives']

                self.remember(prev_state, action, reward, state, done)
                total_reward += reward

                # env.render()
                if done:
                    break

                # DEBUG ONLY
                # if (t % 250 == 0):
                #     print("episode: {} step {}".format(i, t))


            # Close the env (and write monitor result info to disk if it was setup)
            print("EPISODE: {} SCORE: {} TOTAL REWARD {} epsilon {}".format(i, score, total_reward, agent.epsilon))
            env.close()

            if i % 100 == 0 and i > 0:
                training_epsilon = agent.epsilon
                play(self, games=5)
                agent.epsilon = training_epsilon

            if self.memory.__len__() == 1000000:
                self.replay()
                self.memory.clear()
                # Save after training!
                print("Done training! now saving..")
                self.save_weights_to_file(save_file_path + "{}/".format("latest"))
                print("Done training and saving!")




# TODO: ADD REPORTING TO TFBOARD
def play(agent, games=1, game_length=5000, render=False):
    print("Playing {} games:".format(games))
    original_epsilon = agent.epsilon
    agent.epsilon = 0.05
    for game in range(games):
        # Reset
        reset_frame_buffer()
        ob = env.reset()
        state = obs_to_state(ob)
        action = 0
        reward = 0
        done = 0
        total_reward = 0

        # play one game
        for t in range(game_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, info = env.step(action)
            state = obs_to_state(ob)
            total_reward += reward

            if render:
                env.render()
            if done:
                print("game: {} total reward: {}".format(game, total_reward))
                break
        if not done:
            print("game: {} total reward: {}".format(game, total_reward))

    print("Done playing..")
    agent.epsilon = original_epsilon
    return


if __name__ == '__main__':
    # TODO: set args to know if we should train or load from disk and just play according to model
    # TODO: add TensorBoard for logging and reporting. [could be used without TF scoping?]

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env = gym.make('DemonAttack-v0')
    env.seed(0)
    agent = MultiHeadDQNAgent(env.action_space, num_of_agents=7)
    # agent.load_weights_from_file("./MultiHead_DQN_Agent_saves/latest/")
    # epsilon as if we're in episode 200
    agent.epsilon = 1
    # agent.train(episode_count=1500)

    agent.load_weights_from_file(save_file_path+"latest/")
    play(agent, games=5, render=True)
    # else:
    #     train(agent)
    #     play(agent, games=1)
