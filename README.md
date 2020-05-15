# reilr192
Multi-Agent DQN network that plays the Atari game DemonAttack, using TF.

This project was done as a final project for the course Topics in Reinforcement Learning in Ben-Gurion University, and was graded 100.

Each agent is using an identical DQN network, with the last 4 game frames as input and the output is the (expected) Q value for each action.
While training, each agent plays a game on thier own, but all agents are learning from that experience.
Whlie playing, the top performing agents are selected (manually), and in each frame the next action is selected via voting.
