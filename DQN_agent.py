import gym
from gym import wrappers, logger
import tensorflow as tf

# does not inherit from object to make it lighter. could inherit later if needed
class DQNAgent:
    def __init__(self, action_space):
        self.action_space = action_space
        self.epsilon = 1.0 # epsilon changes with "temperture"
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98
        self.batch_size = 150
        self.learning_rate = 0.02
        self.memory = [] # maybe a capped queue?
        self.model = self.create_model()


    # creates the network.
    # if using GPUs probably should be set here
    #      TODO: IMPLEMENT
    def create_model(self):
        return

    def get_action(self, observation, reward, done):
        """epsilon-greedy"""
        if tf.random() > self.epsilon:
            action_vector = self.predict(obs_to_state(observation))
            return tf.arg_max(action_vector)

        else: return self.action_space.sample()

    # add this transition to memory
    #      TODO: IMPLEMENT
    def remember(self, state, action, reward, next_state, done):
        return

    # train model with the new data in memory
    #      TODO: IMPLEMENT
    def replay(self, batch_size):
        return

    #      TODO: IMPLEMENT
    def load_weights_from_file(self, filename):
        return

    #      TODO: IMPLEMENT
    def save_weights_to_file(self, filename):
        return

"""turns observation ob from env to state as used by agent"""
def obs_to_state(ob):
    # TODO: if changing network input, change here
    return ob;

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
        ob = obs_to_state(env.reset())
        state = obs_to_state()

        ## for each episode we play according to agent for (at most) episode_length frames
        ## after X frames, update agent's model using the new data we gathered.
        for t in range(episode_length):
            action = agent.get_action(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            # env.render()
            if done:
                break

            # env.monitor can record video of some episodes. see capped_cubic_video_schedule

            if(t % 1000 == 0):
                agent.replay(agent.batch_size)

        # Train after each episode for when we didn't quite make it to 1000 frames...
        agent.replay(agent.batch_size)

    # Close the env and write monitor result info to disk
    env.close()