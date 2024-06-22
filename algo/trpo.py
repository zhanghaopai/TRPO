import torch
import torch.nn as nn
from model.policy import PolicyNetwork
from model.value import ValueNetwork
from torch.distributions import Categorical



class TRPO(nn.Module):
    def __init__(self,cfg):
        super(TRPO, self).__init__()
        self.cfg=cfg
        # 策略网络
        self.policy_net = PolicyNetwork(cfg.dim_state, cfg.dim_action)
        self.policy_optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=cfg.policy_lr)
        # 价值网络
        self.value_net = ValueNetwork(cfg.dim_state)
        self.value_optimizer = torch.optim.AdamW(self.value_net.parameters(), lr=cfg.value_lr)

    def select_action(self, state):
        state = torch.from_numpy(state).unsqueeze(0)
        #
        action_dist=Categorical(self.policy_net(state))
        return action_dist.sample().item()

    def surrogate_loss(self, log_action_probs, imp_sample_probs, advantages):
        '''
        :param log_action_probs:
        :param imp_sample_probs:
        :param advantages:
        :return:
        '''
        return torch.mean(torch.exp(log_action_probs - imp_sample_probs) * advantages)


    def update(self):
        pass
