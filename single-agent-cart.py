import gym
import torch
import matplotlib.pyplot as plt

from pytorch.DQN import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":
    args = {
        "env_fn": lambda : gym.make("CartPole-v1"),
        "Qnet": FCQ,
        "buffer": ReplayBuffer,

        "net_args": {
            "hidden_layers":(64,64),
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

    agent = Agent(**args)
    agent.train(300)
    print(agent.episode_count)

    plt.plot(agent.rewards)
    # plt.plot(evals)
    plt.show()