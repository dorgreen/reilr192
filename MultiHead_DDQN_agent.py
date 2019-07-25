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
from typing import List
import queue

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
class DDQNAgent:
    def __init__(self, action_space, epsilion=0.95):
        self.action_space = action_space
        self.state_size = WIDTH * HEIGHT * 4
        self.epsilon = epsilion  # epsilon changes with "temperture", resets on each episode. takes about 900 runs
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.gamma = 0.85  # discount
        self.learning_rate = 0.01
        self.model = self.create_model()
        self.target_model = self.create_model()
        self.update_target_model()


    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def huber_loss(self, target, prediction):
        error = prediction - target
        return kr.mean(kr.sqrt(1 + kr.square(error)) - 1, axis=-1)

    # creates the network.
    # if using GPUs probably should be set here
    def create_model(self):

        with tf.device("/cpu:0"):
            inputs = Input(shape=(4, WIDTH, HEIGHT,))
            model = Conv2D(activation='relu', kernel_size=(8, 8), filters=16, strides=(4, 4),
                           padding='same')(inputs)
            model = Conv2D(activation='relu', kernel_size=(4, 4), filters=32, strides=(2, 2),
                           padding='same')(model)
            model = Flatten()(model)
            # Last two layers are fully-connected
            model = Dense(activation='relu', units=256)(model)
            q_values = Dense(activation='linear', units=6)(model)
            m = Model(input=inputs, outputs=q_values)
            m.compile(loss=self.huber_loss,
                      optimizer=Adam(lr=self.learning_rate))

        return m

    def get_action(self, state, reward, done):
        """epsilon-greedy"""
        if random.random() > self.epsilon:
            action_vector = self.model.predict(state)
            return np.argmax(action_vector)

        else:
            return random.randrange(start=0, stop=self.action_space.n)


    # train model with the new data in memory
    #      TODO: add TensorBoard callback on model.predict for metrics report
    def replay(self, batch):
        # update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon = (self.epsilon_decay * self.epsilon)

        self.update_target_model()

        # for each transition in this batch, set target and fit to model accordingly

        for state, action, reward, next_state, done in batch:
            target = reward

            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)))

            target_f = self.target_model.predict(state)
            target_f[0][action] = target
            loss = self.model.fit(state, target_f, epochs=1, verbose=0)

        return

    # TODO: Refactor to save for each agent
    def load_weights_from_file(self, filename):
        self.model.load_weights(filename)

    # TODO: Refactor to save for each agent
    def save_weights_to_file(self, filename):
        self.model.save_weights(filename, True)


def transform_reward(reward, done, lives_delta, episode=-1, action=-1):
    reward = -1 if reward == 0 else 10 * reward
    if episode == -1:
        reward = -30 if lives_delta < 0 else reward
    else:
        reward = max(-30, 0.5 * episode) if lives_delta < 0 else reward
    reward = reward if not done else -20
    return reward


class MultiHeadDQNAgent:
    # TODO: maybe add bootstrap map as a parameter
    def __init__(self, action_space, num_of_agents, epsilon=0.95, agents_epsilon=0.95):
        self.action_space = action_space
        self.agents_num = num_of_agents
        self.state_size = WIDTH * HEIGHT * 4
        self.epsilon = epsilon  # epsilon changes with "temperture", resets on each episode. takes about 900 runs
        self.agents = self.create_model(num_of_agents, agents_epsilon)
        self._frames_buffer = list()
        self.memory = list()


        # Create the empty frame
        temp_env = gym.make('DemonAttack-v0')
        self._emptyframe = self.resize_frame(temp_env.reset())
        temp_env.close()

        # creates the agents.
        # if using GPUs probably should be set here

    def create_model(self, num_of_agents, agents_epsilon) -> List[DDQNAgent]:
        agents = list()
        for i in range(num_of_agents):
            agents.append(DDQNAgent(self.action_space, epsilion=agents_epsilon))

        return agents

    def get_action(self, state, reward, done):
        """epsilon-greedy"""
        # Let all models vote
        # Other options: action with best sum of values, the action with the highest value on some network
        if random.random() > self.epsilon:
            actions = bytearray(6)
            for agent in self.agents:
                ++actions[agent.get_action(state,reward,done)]
            return np.argmax(actions)

        else:
            return random.randrange(start=0, stop=self.action_space.n)

        # add this transition to memory
        #      TODO: Add feature that saves image of state once in a while for debug

    def remember(self, state, action, reward, next_state, done):
        if len(self.memory) >= 2500:
            self.memory.pop()
        self.memory.append((state, action,reward,next_state,done))
        return

        # train model with the new data in memory
        #      TODO: add TensorBoard callback on model.predict for metrics report

    # TODO: IMPLEMET WITH MULTITHREADS!
    def replay(self):
        # mask = np.random.binomial(self.agents_num, p=0.5, size=self.agents_num)
        #
        # for val, ddqn_agent in zip(mask, self.agents):
        #     if val > 0:
        #         self.memory[ddqn_agent].append([state, action, reward, next_state, done])

        # Get a batch for each ddqn agent:
        size = min(len(self.memory), 256)
        for dqn_agent in self.agents:
            batch = random.choices(self.memory,k=size)
            dqn_agent.replay(batch)

        return

    def load_weights_from_file(self, filename: str):
        for i in range(self.agents_num):
            self.agents[i].load_weights_from_file(filename + "{}.h5".format(i))

    def save_weights_to_file(self, filename):
        for i in range(self.agents_num):
            self.agents[i].save_weights_to_file(filename + "{}.h5".format(i))

    def set_agents_epsilon(self, epsilon):
        for agent in self.agents:
            agent.epsilon = epsilon
        return

    @staticmethod
    def resize_frame(ob):
        """turns a single frame from original format to the format in Q function"""
        # TODO: if changing network input, change here
        ## Original obervation is an nd array of (210, 160, 3) , such that H * W * rgb
        return np.array(resize(rgb2gray(ob), (WIDTH, HEIGHT))).flatten()

    def reset_frame_buffer(self):
        self._frames_buffer.clear()
        for i in range(4):
            self._frames_buffer.append(self._emptyframe)
        return

    def obs_to_state(self, ob):
        """turns observation ob from env to state as used by agent"""
        # TODO: if changing network input, change here
        this_frame = self.resize_frame(ob)
        self._frames_buffer.pop(0)  # oldest frame discarded
        self._frames_buffer.append(this_frame)
        return np.array(tuple(self._frames_buffer)).reshape([1, 4, WIDTH, HEIGHT])

    def train(self, episode_count=1000, episode_length=5000):

        ## Gain some experience..
        for i in range(episode_count):
            # Reset items
            self.reset_frame_buffer()
            ob = env.reset()
            state = self.obs_to_state(ob)
            prev_state = state
            action = 0
            reward = 0
            done = False

            total_reward = 0
            score = 0
            lives = 0

            # Select a random agent for this episode
            head_agent = self.agents[random.randrange(start=0, stop=self.agents_num)]

            ## for each episode we play according to head_agent for (at most) episode_length frames
            ## for each agent in self.agents, record each transition acording to it's probability
            for t in range(episode_length):
                action = head_agent.get_action(state, reward, done)
                ob, reward, done, info = env.step(action)
                prev_state = state
                state = self.obs_to_state(ob)
                score += reward
                reward = transform_reward(reward, done, info['ale.lives'] - lives, episode=i, action=action)
                lives = info['ale.lives']

                # TODO: UPDATE TO REMEMBER FOR EACH AGENT!
                agent.remember(prev_state, action, reward, state, done)
                total_reward += reward

                env.render()
                if done:
                    break

                # env.monitor can record video of some episodes. see capped_cubic_video_schedule

                # DEBUG ONLY
                if (t % 250 == 0):
                    print("episode: {} step {}".format(i, t))

                # Train after each episode (for when we didn't quite make it to 2000 frames...)
                agent.replay()

            # Save every 25 runs
            if i % 25 == 0:
                agent.save_weights_to_file("./MultiHead_DQN_Agent_saves/save{}/".format(i))
                agent.save_weights_to_file("./MultiHead_DQN_Agent_saves/latest/")

            # Close the env (and write monitor result info to disk if it was setup)
            print("EPISODE: {} SCORE: {} TOTAL REWARD {} epsilon {}".format(i, score, total_reward, agent.epsilon))
            env.close()

            agent.save_weights_to_file("./MultiHead_DQN_Agent_saves/latest/")
            agent.save_weights_to_file("./MultiHead_DQN_Agent_saves/final/")


def play(agent, games=1, game_length=5000):
    for game in range(games):
        # Reset
        agent.reset_frame_buffer()
        ob = env.reset()
        state = agent.obs_to_state(ob)
        action = 0
        reward = 0
        done = 0
        agent.epsilon = agent.epsilon_min
        total_reward = 0

        # play one game
        for t in range(game_length):
            action = agent.get_action(state, reward, done)
            ob, reward, done, info = env.step(action)
            state = agent.obs_to_state(ob)
            total_reward += reward

            env.render()
            if done:
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
    agent = MultiHeadDQNAgent(env.action_space, num_of_agents=2)
    # agent.load_weights_from_file("./MultiHead_DQN_Agent_saves/save350.h5")
    agent.train()

    agent.load_weights_from_file("./MultiHead_DQN_Agent_saves/lastest.h5")
    agent.epsilon = 0.01
    agent.set_agents_epsilon(0)
    play(agent, games=10)
    # else:
    #     train(agent)
    #     play(agent, games=1)
