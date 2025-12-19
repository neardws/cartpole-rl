import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import DQNAgent
import os


def train(
    env_name="CartPole-v1",
    num_episodes=500,
    max_steps=500,
    render=False,
    save_path="dqn_cartpole.pth",
):
    env = gym.make(env_name, render_mode="human" if render else None)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)

    rewards_history = []
    avg_rewards_history = []

    print(f"Training DQN on {env_name}")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {agent.device}")
    print("-" * 50)

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.learn()

            state = next_state
            episode_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards_history.append(episode_reward)

        avg_reward = np.mean(rewards_history[-100:])
        avg_rewards_history.append(avg_reward)

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.0f} | "
                f"Avg(100): {avg_reward:.1f} | "
                f"Epsilon: {agent.epsilon:.3f}"
            )

        if avg_reward >= 475 and episode >= 100:
            print(f"\nSolved in {episode + 1} episodes!")
            break

    env.close()

    os.makedirs("models", exist_ok=True)
    agent.save(os.path.join("models", save_path))
    print(f"\nModel saved to models/{save_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(rewards_history, alpha=0.6, label="Episode Reward")
    plt.plot(avg_rewards_history, label="Avg Reward (100 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN Training on CartPole-v1")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_curve.png")
    plt.show()
    print("Training curve saved to training_curve.png")

    return agent, rewards_history


if __name__ == "__main__":
    train()
