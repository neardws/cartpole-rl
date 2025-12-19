import gymnasium as gym
import numpy as np
from agent import DQNAgent
import os


def evaluate(
    model_path="models/dqn_cartpole.pth",
    env_name="CartPole-v1",
    num_episodes=10,
    render=True,
):
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    else:
        print(f"Model not found at {model_path}, using untrained agent")

    rewards = []
    print(f"\nEvaluating on {env_name} for {num_episodes} episodes...")
    print("-" * 50)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0

        while True:
            action = agent.select_action(state, training=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            state = next_state
            episode_reward += reward
            steps += 1

            if done:
                break

        rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}, Steps = {steps}")

    env.close()

    print("-" * 50)
    print(f"Average Reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})")
    print(f"Max Reward: {np.max(rewards):.0f}")
    print(f"Min Reward: {np.min(rewards):.0f}")

    return rewards


if __name__ == "__main__":
    evaluate()
