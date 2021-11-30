import random

import gym
import numpy as np
from IPython.display import clear_output

from gym.envs.atari.environment import AtariEnv

# Create the Ms. Pacman environment
env: AtariEnv = gym.make('MsPacman-v0', render_mode="human").env
print(env.get_action_meanings())

# Initialize the Q-Table
state_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n
q_table: np.ndarray = np.zeros([state_space_size, action_space_size])

# Set Q-Learning parameters
ALPHA: float = 0.1  # Learning rate
GAMMA: float = 0.6  # Discount factor; determines importance of future rewards
EPSILON: float = 0.1  # Determines the frequency at which new states are explored
NUM_EPISODES: int = 10000  # The number of episodes to train the agent

all_epochs: list = []
all_penalties: list = []

# Train the agent using Q-Learning
for i in range(NUM_EPISODES):
    # Reset the state before each epoch
    state: np.ndarray = env.reset()

    epochs: int = 0
    penalties: int = 0
    reward: int = 0
    done: bool = False

    while not done:
        # Decide between exploration and exploitation
        if random.uniform(0, 1) < EPSILON:
            # Choose a random action sometimes
            action = env.action_space.sample()
        else:
            # Choose the action for optimal reward
            action = np.argmax(q_table[state])

        # Execute the selected action
        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        # Update the Q-Table
        new_value = (1 - ALPHA) * old_value + ALPHA * (reward + GAMMA * next_max)
        q_table[state, action] = new_value

        if reward == 10:
            penalties += 1

        state = next_state
        epochs += 1

    if i % 100 == 0:
        clear_output(wait=True)
        print(f"Episode: {i}")


env.close()
print("Training finished")
