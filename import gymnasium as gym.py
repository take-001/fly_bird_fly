import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import random

# Initialize the environment with lidar observations and RGB rendering
env = gym.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=True)

# Parameters
state_space_bins = 10  # Number of bins for discretizing continuous state space
learning_rate = 0.1
discount_factor = 0.99
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995
epsilon_min = 0.01
episodes = 5000

# Discretize state space
def discretize_state(state, bins):
    """
    Discretize the continuous lidar state into bins for Q-learning.
    The lidar observations are already structured, so binning simplifies it.
    """
    return tuple(int(np.digitize(s, np.linspace(-1, 1, bins))) for s in state)

# Initialize Q-table
# Assuming the lidar provides a fixed number of observations (e.g., 10 beams)
lidar_dims = len(env.observation_space.sample())  # Adjust for actual lidar dimensions
q_table = np.zeros((state_space_bins,) * lidar_dims + (2,), dtype=np.uint32)  # 2 actions: flap or no-flap

# Training loop
for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state, state_space_bins)
    total_reward = 0
    done = False
    
    while not done:
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit
        
        # Perform the action
        next_state, reward, done, _, _ = env.step(action)
        next_state = discretize_state(next_state, state_space_bins)
        total_reward += reward

        # Update Q-table using Bellman equation
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += learning_rate * (
            reward + discount_factor * q_table[next_state][best_next_action] - q_table[state][action]
        )

        # Move to the next state
        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

# Save the trained Q-table
np.save("q_table_flappy_bird_lidar.npy", q_table)

env.close()
print("Training complete!")