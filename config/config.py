import configparser


class Config:
    def __init__(self, filename):
        config = configparser.ConfigParser()
        config.read(filename, encoding="gbk")
        self.env_name = config.get("Train", "env_name")  # 环境名称
        self.train_epoch = config.getint("Train", "epoch")  # 环境名称
        self.rollout_num=config.getint("Train", "rollout_num") #玩到底的次数
        self.discount=config.getfloat("Train", "discount") #折扣因子
        self.value_lr=config.getfloat("Train", "value_lr") # 学习率
        self.policy_lr=config.getfloat("Train", "policy_lr") # 学习率
        self.sample_num=config.getint("Train", "sample_num") # 学习率
        self.delta=config.getfloat("Train", "delta")

