import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from pytorch.Federator import Federator
from pytorch.DQN import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

if __name__ == "__main__":

    args = {
        "env_fn": lambda : gym.make("CartPole-v1"),
        "Qnet": FCQ,
        "buffer": ReplayBuffer,

        "net_args" : {
            "hidden_layers":(512, 128),
            "activation_fn":torch.nn.functional.relu,
            "optimizer":torch.optim.Adam, # torch.optim.RMSprop
            "learning_rate":0.0005,
        },

        "max_epsilon": 1.0,
        "min_epsilon": 0.1,
        "decay_steps": 2000,
        "gamma": 0.99,
        "target_update_rate": 15,
        "min_buffer": 64
    }

    n_runs = 2000
    n_agents = 3
    n_iterations = 10
    update_rate = 30

    fed_rewards = np.zeros(n_runs)
    for i in range(n_iterations):
        fed = Federator(n_agents=n_agents, update_rate=update_rate, args=args)
        fed_rewards += fed.train(n_runs)
    fed_rewards /= n_iterations
    fed.print_episode_lengths()
    with open('fed_rewards.npy', 'wb') as f:
        np.save(f, fed_rewards)

    
    single_rewards = np.zeros(n_runs)
    for i in range(n_iterations):
        ag = Agent(**args)
        for r in tqdm(range(n_runs)):
            ag.step(update_rate)
            single_rewards[r] += ag.evaluate()
    single_rewards /= n_iterations
    with open('single_rewards.npy', 'wb') as f:
        np.save(f, single_rewards)

    plt.plot(fed_rewards, color="b", label="federated")
    plt.plot(single_rewards, color="r", label="single")
    plt.legend()
    plt.show()
    