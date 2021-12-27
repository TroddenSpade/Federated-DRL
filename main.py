import gym
import torch

from pytorch.Agent import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":
    net_args = {
        "hidden_layers":(32,64),
        "activation_fn":torch.nn.functional.relu,
        "optimizer":torch.optim.Adam,
        "learning_rate":0.0005,
    }

    args = {
        "env_fn": lambda : gym.make("CartPole-v1"),
        "Qnet": lambda : FCQ(**net_args),
        "buffer": ReplayBuffer,

        "epsilon": 0.1,
        "target_update_rate": 0.1
    }

    
    agent = Agent(**args)