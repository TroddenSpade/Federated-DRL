import gym
import torch
import matplotlib.pyplot as plt

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
        "Qnet": FCQ,
        "buffer": ReplayBuffer,

        "epsilon": 0.1,
        "gamma": 1,
        "target_update_rate": 15,
        "min_buffer": 64
    }

    agent = Agent(**args)
    rewards, evals = agent.train(300)

    plt.plot(rewards)
    plt.plot(evals)
    plt.show()