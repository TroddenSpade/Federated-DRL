import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch.Federator import Federator
from pytorch.Agent import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":
    N_AGENTS = 3

    net_args = {
        "hidden_layers":(32,64),
        "activation_fn":torch.nn.functional.relu,
        "optimizer":torch.optim.Adam,
        "learning_rate":0.001,
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

    # fed = Federator(n_agents=5, update_rate=50, args=args)
    # rewards = fed.train(400)

    ag = Agent(**args)
    rewards_ = np.zeros(400)
    for r in tqdm(range(400)):
        ag.step(50)
        rewards_[r] = ag.evaluate()

    print(ag.episode_count)
    # plt.plot(rewards, color="b")
    plt.plot(rewards_, color="r")
    # plt.legend()
    plt.show()
    