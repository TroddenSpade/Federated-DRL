import numpy as np
from tqdm import tqdm

from pytorch.DQN import Agent
from pytorch.QNetwork import FCQ
from pytorch.ReplayBuffer import ReplayBuffer

class Federator:
    def __init__(self, n_agents, update_rate, args) -> None:
        
        self.env = args["env_fn"]()
        self.n_actions = self.env.action_space.n
        self.state_shape = self.env.observation_space.shape

        self.global_agent = Agent(**args)

        self.update_rate = update_rate
        self.n_agents = n_agents
        self.agents = []
        for _ in range(n_agents):
            agent = Agent(**args)
            self.agents.append(agent)

        self.set_local_networks()


    def print_episode_lengths(self):
        for a in self.agents:
            print(a.episode_count)
            
    def train(self, n_runs):
        rewards = np.zeros(n_runs)
        for r in tqdm(range(n_runs)):
            scores = []
            for agent in self.agents:
                agent.step(self.update_rate)
                scores.append(agent.get_score())
            self.aggregate_networks(scores)
            self.set_local_networks()
            rewards[r] = self.global_agent.evaluate()
        return rewards


    def aggregate_networks(self, scores):
        sd_online = self.global_agent.online_net.state_dict()
        sd_target = self.global_agent.target_net.state_dict()

        online_dicts = []
        target_dicts = []
        for agent in self.agents:
            online_dicts.append(agent.online_net.state_dict())
            target_dicts.append(agent.target_net.state_dict())

        for key in sd_online:
            sd_online[key] -= sd_online[key]
            for i, dict in enumerate(online_dicts):
                sd_online[key] += scores[i] * dict[key]
            sd_online[key] /= sum(scores)

        for key in sd_target:
            sd_target[key] -= sd_target[key]
            for i, dict in enumerate(target_dicts):
                sd_target[key] += scores[i] * dict[key]
            sd_target[key] /= sum(scores)

        self.global_agent.online_net.load_state_dict(sd_online)
        self.global_agent.target_net.load_state_dict(sd_target)


    def set_local_networks(self):
        for agent in self.agents:
            agent.online_net.load_state_dict(
                self.global_agent.online_net.state_dict())
            agent.target_net.load_state_dict(
                self.global_agent.target_net.state_dict())