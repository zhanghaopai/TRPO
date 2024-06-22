import torch.nn.functional as F
import numpy as np

# 计算KL散度
def kl_divergence(old_policy, new_policy, states):
    # 原来的概率分布
    old_probs = old_policy(states)
    # 新的概率分布
    new_probs = new_policy(states)
    # KL(/pi_{/theta_{old}} || /pi_{/theta})：看新策略的分布对老策略的分布之间的距离
    return F.kl_div(input=new_probs.log(), target=old_probs, reduction='sum')




def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards, dtype=np.float32)
    running_add = 0
    for t in reversed(range(len(rewards))):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards
