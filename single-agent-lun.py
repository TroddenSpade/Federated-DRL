import gym
import torch
import matplotlib.pyplot as plt
import numpy as np

from pytorch.DQN import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":
    args = {
        "env_fn": lambda : gym.make("LunarLander-v2"),
        "Qnet": FCQ,
        "buffer": ReplayBuffer,

        "net_args": {
            "hidden_layers":(512, 256, 128),
            "activation_fn":torch.nn.functional.relu,
            "optimizer":torch.optim.Adam,
            "learning_rate":0.0005,
        },

        "max_epsilon": 1.0,
        "min_epsilon": 0.1,
        "decay_steps": 5000,
        "gamma": 0.99,
        "target_update_rate": 15,
        "min_buffer": 64
    }

    rewards = np.zeros(200)
    for i in range(10):
        agent = Agent(**args)
        agent.train(200)
        print(agent.step_count)
        rewards += agent.rewards

    plt.plot(rewards/10)
    # plt.plot(evals)
    plt.show()