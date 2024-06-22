import logging

import gym
import torch
import numpy as np

from algo.trpo import TRPO
from config.config import Config
from model.policy import PolicyNetwork
from utils.trajectory import Trajectory
from utils.utils import *

logging.basicConfig(level=logging.DEBUG)


def train(cfg, agent, env):
    for episode in range(cfg.train_epoch):
        state = env.reset()
        state = state[0]
        trajectory = Trajectory()
        # 抽样sample_num次
        log_probs = []
        for t in range(cfg.sample_num):
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs = agent.policy_net(state)
            logging.debug("action概率分布为：%s", action_probs.tolist())
            # 抽样
            action = torch.multinomial(input=action_probs, num_samples=1).item()
            logging.debug("抽样1次action样本为：%s", action)
            # 下一次的状态，奖励，该回合是否结束，一些额外信息info
            # action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(np.array([action]))
            trajectory.add(state, action, reward, done, next_state)
            log_prob = torch.log(action_probs.squeeze(0)[action])
            log_probs.append(log_prob)
            state = next_state
            if done:
                break
        # 根据抽样更新策略网络
        states = torch.cat(trajectory.state)
        actions = torch.tensor(trajectory.action)
        rewards = discount_rewards(trajectory.reward, 0.1)
        rewards = torch.FloatTensor(rewards)
        logging.debug("states分布：%s", states)
        logging.debug(actions)


        values = agent.value_net(states).squeeze()
        advantages = rewards - values.detach()

        # 记住当前的网络
        old_policy = PolicyNetwork(cfg.dim_state, cfg.dim_action)
        old_policy.load_state_dict(agent.policy_net.state_dict())
        # 近似
        for _ in range(10):
            new_log_probs = torch.log(
                agent.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze())
            ratio = torch.exp(new_log_probs - torch.tensor(log_probs))

            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - cfg.delta, 1.0 + cfg.delta) * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            policy_loss.backward()
            agent.policy_optimizer.step()

            kl = kl_divergence(old_policy, agent.policy_net, states)
            if kl > 1.5 * cfg.delta:
                print(f"KL divergence too high ({kl:.4f}). Stopping training.")
                break


def test(cfg, agent, env):
    for epoch in range(cfg.train_epoch):
        pass


def create_env(cfg, mode):
    '''
    初始化游戏环境
    :param cfg: 配置，主要获取游戏名称
    :param mode: 游戏方式，区分human可视化，和rgb非可视化
    :return: 返回游戏环境、状态空间大小、动作空间大小
    '''
    env = gym.make(cfg.env_name, render_mode=mode)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]
    return env, dim_state, dim_action


def main():
    cfg = Config("config/config.ini")
    env, dim_state, dim_action = create_env(cfg, "human")
    cfg.dim_state = dim_state
    cfg.dim_action = dim_action

    agent = TRPO(cfg)
    train(cfg, agent, env)
    # next_state, reward, done, info = env.step(action)


if __name__ == "__main__":
    main()
