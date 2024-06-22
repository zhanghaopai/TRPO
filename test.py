import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# 定义价值网络
class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# 计算KL散度
def kl_divergence(old_policy, new_policy, states):
    old_probs = old_policy(states)
    new_probs = new_policy(states)
    kl = old_probs * torch.log(old_probs / (new_probs + 1e-10))
    return kl.sum(dim=1).mean()


# 计算折扣奖励
def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards


# 参数设置
env_name = "CartPole-v1"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNetwork(state_dim, action_dim).to(device)
value = ValueNetwork(state_dim).to(device)
policy_optimizer = optim.Adam(policy.parameters(), lr=1e-3)
value_optimizer = optim.Adam(value.parameters(), lr=1e-3)

max_episodes = 1000
max_timesteps = 200
gamma = 0.99
gae_lambda = 0.95
delta = 0.01

for episode in range(max_episodes):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    log_probs = []

    for t in range(max_timesteps):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action_probs = policy(state)
        action = torch.multinomial(action_probs, 1).item()

        next_state, reward, done, _ = env.step(action)

        log_prob = torch.log(action_probs.squeeze(0)[action])
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)

        state = next_state
        if done:
            break

    # 更新策略网络
    states = torch.cat(states)
    actions = torch.tensor(actions).to(device)
    rewards = discount_rewards(rewards, gamma)
    rewards = torch.FloatTensor(rewards).to(device)

    values = value(states).squeeze()
    advantages = rewards - values.detach()

    old_policy = PolicyNetwork(state_dim, action_dim).to(device)
    old_policy.load_state_dict(policy.state_dict())

    for _ in range(10):
        new_log_probs = torch.log(policy(states).gather(1, actions.unsqueeze(1)).squeeze())
        ratio = torch.exp(new_log_probs - torch.cat(log_probs))

        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1.0 - delta, 1.0 + delta) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        kl = kl_divergence(old_policy, policy, states)
        if kl > 1.5 * delta:
            print(f"KL divergence too high ({kl:.4f}). Stopping training.")
            break

    # 更新价值网络
    value_loss = ((rewards - values) ** 2).mean()
    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    print(f"Episode {episode + 1}/{max_episodes}, Reward: {sum(rewards):.2f}")

env.close()
