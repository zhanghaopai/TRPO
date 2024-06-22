import gym
import torch
import torch.nn as nn
import torch.optim as optim


# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)


# 初始化环境
env = gym.make('CartPole-v1', render_mode="human")

# 初始化网络和优化器
policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
optimizer = optim.Adam(policy_net.parameters())
num_episodes=1000
# 训练循环
for episode in range(num_episodes):
    states, actions, rewards = [], [], []
    state = env.reset()
    done = False

    while not done:
        # 前向传播，获取动作概率
        print(type(torch.from_numpy(state).float().unsqueeze(0)))
        probs = policy_net(torch.from_numpy(state).float().unsqueeze(0))
        # 根据概率选择动作
        action = torch.distributions.Categorical(probs).sample()
        # 执行动作，获取下一个状态和奖励
        next_state, reward, done, _ = env.step(action.item())
        # 收集数据
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    # 这里需要实现计算优势函数、策略梯度、Fisher矩阵等步骤
    # ...

    # 更新策略网络
    optimizer.step()
    optimizer.zero_grad()

# 评估策略性能